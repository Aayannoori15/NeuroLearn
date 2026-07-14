import uuid
import re as _re
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from schemas import (
    StartSessionRequest,
    StartSessionResponse,
    DiagnosticRequest,
    DiagnosticResponse,
    GenerateRequest,
    LessonResponse,
    ExerciseResponse,
    SubmitExerciseRequest,
    SubmitExerciseResponse,
    ProgressResponse,
    WeaknessProfileResponse,
    MaterialUploadResponse,
    MaterialGenerateRequest,
    MaterialLessonResponse,
    MaterialExerciseResponse,
    FlashcardRequest,
    FlashcardResponse,
    PodcastRequest,
    PodcastResponse,
    UserCreate,
    UserLogin,
    User,
    Token,
)
from database import create_session, get_session, update_session, get_users_collection
from slowapi import Limiter
from slowapi.util import get_remote_address
from jose import jwt, JWTError
from auth import create_access_token, get_password_hash, verify_password, get_current_user, SECRET_KEY, ALGORITHM
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends
from adaptive_engine import (
    calculate_level,
    adjust_level,
    generate_lesson_prompt,
    generate_exercise_prompt,
    generate_diagnostic_prompt,
)
from gemini_client import generate_text, generate_json
from performance_tracker import (
    record_answers,
    compute_mastery,
    detect_weaknesses,
    get_study_recommendations,
    detect_stress,
    get_weakness_dna,
    empty_performance,
    suggest_next_topic,
)
from material_rag import (
    extract_text,
    chunk_text,
    store_chunks,
    retrieve_chunks,
    has_material,
    build_rag_lesson_prompt,
    build_rag_exercise_prompt,
)
from flashcard_engine import (
    generate_flashcard_prompt,
    generate_flashcard_custom_topic_prompt,
    generate_flashcard_from_material_prompt,
    validate_flashcards,
)

def _rate_key(request: Request) -> str:
    """
    Rate-limit key: user_id from JWT when authenticated, otherwise remote IP.
    Gives per-user limits on content endpoints, per-IP for public auth endpoints.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            uid = payload.get("id")
            if uid:
                return f"user:{uid}"
        except (JWTError, Exception):
            pass
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=_rate_key)
router  = APIRouter()


def _ownership_check(session: dict, current_user: dict) -> None:
    """Raise 403 if this session belongs to a different authenticated user."""
    session_uid = session.get("user_id")
    if session_uid and session_uid != current_user.get("user_id"):
        raise HTTPException(status_code=403, detail="Access denied to this session")


# --- Authentication ---
@router.post("/auth/register", response_model=User)
@limiter.limit("10/minute")
async def register(request: Request, user: UserCreate):
    users_collection = get_users_collection()
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = {
        "user_id": str(uuid.uuid4()),
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_password,
        "is_active": True,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    await users_collection.insert_one(new_user)
    
    return User(
        id=new_user["user_id"],
        username=new_user["username"],
        email=new_user["email"],
        is_active=new_user["is_active"]
    )

@router.post("/auth/login", response_model=Token)
@limiter.limit("5/minute")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    users_collection = get_users_collection()
    user = await users_collection.find_one({"email": form_data.username}) # OAuth2 form uses 'username' field for email usually
    
    if not user:
        # Fallback: check if username matches
        user = await users_collection.find_one({"username": form_data.username})
    
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"], "id": user["user_id"]})
    return {"access_token": access_token, "token_type": "bearer", "userId": user["user_id"]}

# --- Existing routes ---


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@router.post("/start-session", response_model=StartSessionResponse)
@limiter.limit("20/minute")
async def start_session(request: Request, req: StartSessionRequest, current_user: dict = Depends(get_current_user)):
    session_id = uuid.uuid4().hex[:12]
    await create_session(session_id, req.subject, user_id=current_user["user_id"])
    return StartSessionResponse(session_id=session_id, subject=req.subject, level="unknown")


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

@router.post("/diagnostic-questions")
@limiter.limit("10/minute")
async def diagnostic_questions(request: Request, req: GenerateRequest, current_user: dict = Depends(get_current_user)):
    session = await get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _ownership_check(session, current_user)

    prompt = generate_diagnostic_prompt(session["subject"], req.question_type)
    questions = await generate_json(prompt, task="diagnostic")
    return {"questions": questions, "subject": session["subject"]}


@router.post("/diagnostic", response_model=DiagnosticResponse)
@limiter.limit("10/minute")
async def diagnostic(request: Request, req: DiagnosticRequest, current_user: dict = Depends(get_current_user)):
    session = await get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _ownership_check(session, current_user)

    correct, total = _score_answers(req.answers)
    score = (correct / total * 100) if total > 0 else 0
    level = calculate_level(score)

    # Update performance tracker
    perf = session.get("performance", None)
    if perf is None:
        from performance_tracker import empty_performance
        perf = empty_performance()
    scored = _add_correct_flags(req.answers)
    qtype = scored[0].get("type", "short") if scored else "short"
    perf = record_answers(perf, scored, qtype, session["subject"])

    history = session["level_history"] + [level]
    await update_session(
        session["id"],
        level=level,
        total_correct=session["total_correct"] + correct,
        total_attempts=session["total_attempts"] + total,
        level_history=history,
        performance=perf,
    )

    return DiagnosticResponse(score=round(score, 1), level=level, correct=correct, total=total)


# ---------------------------------------------------------------------------
# Lesson & Exercise
# ---------------------------------------------------------------------------

@router.post("/generate-lesson", response_model=LessonResponse)
@limiter.limit("10/minute")
async def generate_lesson(request: Request, req: GenerateRequest, current_user: dict = Depends(get_current_user)):
    session = await get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["level"] == "unknown":
        raise HTTPException(status_code=400, detail="Complete diagnostic first")
    _ownership_check(session, current_user)

    prompt = generate_lesson_prompt(session["subject"], session["level"])
    lesson_text = await generate_text(prompt, task="lesson")
    return LessonResponse(lesson=lesson_text, subject=session["subject"], level=session["level"])


@router.post("/generate-exercise", response_model=ExerciseResponse)
@limiter.limit("10/minute")
async def generate_exercise(request: Request, req: GenerateRequest, current_user: dict = Depends(get_current_user)):
    session = await get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["level"] == "unknown":
        raise HTTPException(status_code=400, detail="Complete diagnostic first")
    _ownership_check(session, current_user)

    prompt = generate_exercise_prompt(session["subject"], session["level"], req.question_type)
    questions = await generate_json(prompt, task="exercise")
    return ExerciseResponse(questions=questions, subject=session["subject"], level=session["level"])


@router.post("/submit-exercise", response_model=SubmitExerciseResponse)
@limiter.limit("20/minute")
async def submit_exercise(request: Request, req: SubmitExerciseRequest, current_user: dict = Depends(get_current_user)):
    session = await get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _ownership_check(session, current_user)

    correct, total = _score_answers(req.answers)
    accuracy = (correct / total * 100) if total > 0 else 0

    # Resolve timing data
    per_q_times: list[float] | None = None
    total_time = 0.0
    if req.per_question_times and len(req.per_question_times) == total:
        per_q_times = [max(0.0, float(t)) for t in req.per_question_times]
        total_time = sum(per_q_times)
    elif req.total_time_seconds is not None:
        total_time = max(0.0, float(req.total_time_seconds))

    # Update performance
    perf = session.get("performance") or empty_performance()
    scored = _add_correct_flags(req.answers)
    qtype = scored[0].get("type", "short") if scored else "short"
    perf = record_answers(
        perf, scored, qtype, session["subject"],
        time_seconds=total_time,
        per_question_times=per_q_times,
    )
    mastery = compute_mastery(perf)

    new_level = adjust_level(session["level"], accuracy, mastery)
    level_changed = new_level != session["level"]

    history = session["level_history"]
    if level_changed:
        history = history + [new_level]

    await update_session(
        session["id"],
        level=new_level,
        total_correct=session["total_correct"] + correct,
        total_attempts=session["total_attempts"] + total,
        level_history=history,
        performance=perf,
    )

    # Cognitive metrics
    rt_list: list[float] = perf.get("response_times", [])
    avg_rt = round(sum(rt_list) / len(rt_list), 1) if rt_list else 0.0
    csi = perf.get("cognitive_strain_index", 0.0)
    adaptive_mode = perf.get("adaptive_mode", "standard") or "standard"

    # Stress detection
    stress_signal = detect_stress(perf)

    return SubmitExerciseResponse(
        accuracy=round(accuracy, 1),
        correct=correct,
        total=total,
        new_level=new_level,
        level_changed=level_changed,
        mastery=round(mastery, 1),
        adaptive_mode=adaptive_mode,
        cognitive_strain_index=csi,
        avg_response_time=avg_rt,
        stress_detected=stress_signal["stress_detected"],
        recommended_action=stress_signal["recommended_action"],
    )


# ---------------------------------------------------------------------------
# Material upload (RAG)
# ---------------------------------------------------------------------------

@router.post("/upload-material", response_model=MaterialUploadResponse)
@limiter.limit("5/minute")
async def upload_material(
    request: Request,
    session_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    # Session lookup is optional — allows standalone uploads without picking a subject
    session = await get_session(session_id)
    if session:
        _ownership_check(session, current_user)

    content = await file.read()
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "pptx"):
        raise HTTPException(status_code=400, detail="Only PDF and PPTX files are supported")

    import io as _io
    text = extract_text(filename, _io.BytesIO(content))
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    chunks = chunk_text(text)
    store_chunks(session_id, chunks, filename=filename)

    return MaterialUploadResponse(
        session_id=session_id,
        filename=filename,
        chunks=len(chunks),
        message=f"Processed {len(chunks)} chunks from {filename}",
    )


@router.post("/generate-from-material")
@limiter.limit("10/minute")
async def generate_from_material(request: Request, req: MaterialGenerateRequest, current_user: dict = Depends(get_current_user)):
    # Try session lookup first; fall back to request-level subject/level
    session = await get_session(req.session_id)
    if session:
        _ownership_check(session, current_user)
    if not has_material(req.session_id):
        raise HTTPException(status_code=400, detail="No material uploaded for this session")

    subject = (session["subject"] if session else None) or req.subject or "General"
    level = (session.get("level") if session else None) or req.level or "Beginner"
    if level == "unknown":
        level = "Beginner"

    query = f"{subject} {level}"
    chunks = retrieve_chunks(req.session_id, query, top_k=5)

    if req.mode == "lesson":
        prompt = build_rag_lesson_prompt(chunks, subject, level)
        text = await generate_text(prompt, task="lesson")
        return MaterialLessonResponse(lesson=text, source="uploaded material")
    else:
        prompt = build_rag_exercise_prompt(chunks, subject, level, req.question_type)
        questions = await generate_json(prompt, task="exercise")
        return MaterialExerciseResponse(questions=questions, source="uploaded material")


# ---------------------------------------------------------------------------
# Flashcards
# ---------------------------------------------------------------------------

@router.post("/generate-flashcards", response_model=FlashcardResponse)
@limiter.limit("10/minute")
async def generate_flashcards(request: Request, req: FlashcardRequest, current_user: dict = Depends(get_current_user)):
    # Try session lookup; fall back to request-level subject/level for standalone
    session = await get_session(req.session_id) if req.session_id else None
    if session:
        _ownership_check(session, current_user)

    subject = (session["subject"] if session else None) or req.subject
    level = (session.get("level") if session else None) or req.level or "Beginner"
    if level == "unknown":
        level = "Beginner"

    # Resolve custom_topic: allow the old `topic` field to act as custom_topic
    # when no subject is available (backward compat)
    custom_topic = (req.custom_topic or "").strip()
    topic = (req.topic or "").strip()

    # Determine generation mode:
    #   1. from_material → RAG
    #   2. custom_topic (or topic when no subject) → direct LLM with free-form topic
    #   3. subject-based (optionally focused by topic)
    if req.from_material and req.session_id and has_material(req.session_id):
        query = f"{subject or ''} {topic or custom_topic}".strip()
        chunks = retrieve_chunks(req.session_id, query, top_k=5)
        prompt = generate_flashcard_from_material_prompt(chunks)
        display_subject = subject or custom_topic or "Uploaded Material"
    elif custom_topic:
        prompt = generate_flashcard_custom_topic_prompt(custom_topic)
        display_subject = custom_topic
    elif subject:
        prompt = generate_flashcard_prompt(subject, level, topic or None)
        display_subject = subject
    elif topic:
        # topic provided but no subject — treat as custom topic
        prompt = generate_flashcard_custom_topic_prompt(topic)
        display_subject = topic
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide a subject, custom_topic, or topic to generate flashcards.",
        )

    # Generate with one retry on empty/invalid result
    cards = await generate_json(prompt, task="flashcard")
    normalized = validate_flashcards(cards)

    if len(normalized) < 1:
        # Retry once
        cards = await generate_json(prompt, task="flashcard", retries=1)
        normalized = validate_flashcards(cards)

    if not normalized:
        raise HTTPException(
            status_code=502,
            detail="AI failed to generate valid flashcards. Please try again.",
        )

    return FlashcardResponse(flashcards=normalized, subject=display_subject)


# ---------------------------------------------------------------------------
# Progress / Dashboard
# ---------------------------------------------------------------------------

@router.post("/progress", response_model=ProgressResponse)
@limiter.limit("30/minute")
async def progress(request: Request, req: GenerateRequest, current_user: dict = Depends(get_current_user)):
    session = await get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _ownership_check(session, current_user)

    total = session["total_attempts"]
    accuracy = (session["total_correct"] / total * 100) if total > 0 else 0

    perf = session.get("performance") or empty_performance()

    mastery = compute_mastery(perf)
    weaknesses = detect_weaknesses(perf)
    recs = get_study_recommendations(perf, session["subject"])

    # Suggest the weakest topic from those already attempted
    attempted_topics = list(perf.get("topic_accuracy", {}).keys())
    suggested_topic = suggest_next_topic(perf, attempted_topics) if attempted_topics else None

    rt_list: list[float] = perf.get("response_times", [])
    avg_rt = round(sum(rt_list) / len(rt_list), 1) if rt_list else 0.0

    return ProgressResponse(
        session_id=session["id"],
        subject=session["subject"],
        level=session["level"],
        total_correct=session["total_correct"],
        total_attempts=total,
        accuracy=round(accuracy, 1),
        level_history=session["level_history"],
        mastery=round(mastery, 1),
        weaknesses=weaknesses,
        recommendations=recs,
        topic_accuracy=perf.get("topic_accuracy", {}),
        type_accuracy=perf.get("type_accuracy", {}),
        cognitive_strain_index=perf.get("cognitive_strain_index", 0.0),
        avg_response_time=avg_rt,
        adaptive_mode=perf.get("adaptive_mode"),
        weakness_profile=get_weakness_dna(perf),
        suggested_topic=suggested_topic,
    )


@router.get("/weakness-profile/{session_id}", response_model=WeaknessProfileResponse)
@limiter.limit("30/minute")
async def weakness_profile(request: Request, session_id: str, current_user: dict = Depends(get_current_user)):
    """Return the Weakness DNA profile for a session."""
    session = await get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _ownership_check(session, current_user)

    perf = session.get("performance") or empty_performance()
    return WeaknessProfileResponse(
        session_id=session["id"],
        subject=session["subject"],
        weakness_profile=get_weakness_dna(perf),
    )


# ---------------------------------------------------------------------------
# Podcast
# ---------------------------------------------------------------------------

@router.post("/generate-podcast", response_model=PodcastResponse)
@limiter.limit("3/minute")
async def generate_podcast_route(request: Request, req: PodcastRequest, current_user: dict = Depends(get_current_user)):
    from podcast_engine import create_podcast
    result = await create_podcast(req.topic)
    return PodcastResponse(**result)


@router.get("/podcast-audio/{filename}")
@limiter.limit("60/minute")
async def serve_podcast_audio(request: Request, filename: str):
    """Serve generated podcast audio files (mp3/wav)."""
    from fastapi.responses import FileResponse
    from podcast_engine import PODCAST_DIR

    if not _re.match(r"^[a-z0-9_]+\.(mp3|wav)$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = PODCAST_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    media = "audio/mpeg" if filename.endswith(".mp3") else "audio/wav"
    return FileResponse(
        path,
        media_type=media,
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_answer_correct(qtype: str, user: str, expected: str) -> bool:
    """Check if a single answer is correct based on question type.

    - true_false : normalises true/1/yes vs false/0/no
    - qa         : token Jaccard similarity >= 0.30 after stop-word removal
    - default    : exact string match (MCQ, short answer)
    """
    if qtype == "true_false":
        if user in ("true", "1", "yes") and expected in ("true", "1", "yes"):
            return True
        if user in ("false", "0", "no") and expected in ("false", "0", "no"):
            return True
        return False
    elif qtype == "qa":
        if not user or not expected:
            return False
        _STOP = {
            "the", "a", "an", "is", "it", "in", "of", "to", "and", "or", "that",
            "this", "are", "was", "be", "for", "on", "with", "as", "at", "by",
            "from", "has", "its", "but", "not", "have", "had",
        }
        u_words = set(user.split()) - _STOP
        e_words = set(expected.split()) - _STOP
        # If stop-word removal empties both sides, fall back to exact match
        if not u_words or not e_words:
            return user == expected
        intersection = u_words & e_words
        union        = u_words | e_words
        return len(intersection) / len(union) >= 0.30
    else:
        return user == expected



def _add_correct_flags(answers: list[dict]) -> list[dict]:
    """Return answer dicts with a 'correct' boolean added, based on scoring logic."""
    scored = []
    for ans in answers:
        qtype = ans.get("type", "short")
        user = str(ans.get("user_answer", "")).strip().lower()
        expected = str(ans.get("correct_answer", "")).strip().lower()
        scored.append({**ans, "correct": _is_answer_correct(qtype, user, expected)})
    return scored


def _score_answers(answers: list[dict]) -> tuple[int, int]:
    """Return (correct, total) from a list of answer dicts."""
    correct = 0
    total = len(answers)
    for ans in answers:
        qtype = ans.get("type", "short")
        user = str(ans.get("user_answer", "")).strip().lower()
        expected = str(ans.get("correct_answer", "")).strip().lower()
        if _is_answer_correct(qtype, user, expected):
            correct += 1
    return correct, total
