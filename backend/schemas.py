from pydantic import BaseModel
from typing import Optional


class StartSessionRequest(BaseModel):
    subject: str


class StartSessionResponse(BaseModel):
    session_id: str
    subject: str
    level: str


class DiagnosticRequest(BaseModel):
    session_id: str
    answers: list[dict]  # [{question, user_answer, correct_answer, type?}]


class DiagnosticResponse(BaseModel):
    score: float
    level: str
    correct: int
    total: int


class GenerateRequest(BaseModel):
    session_id: str
    question_type: str = "short"  # mcq | true_false | short | qa | mixed


class LessonResponse(BaseModel):
    lesson: str
    subject: str
    level: str


class ExerciseResponse(BaseModel):
    questions: list[dict]
    subject: str
    level: str


class SubmitExerciseRequest(BaseModel):
    session_id: str
    answers: list[dict]  # [{question, user_answer, correct_answer, type?}]
    per_question_times: Optional[list[float]] = None  # seconds per question
    total_time_seconds: Optional[float] = None        # fallback total time


class SubmitExerciseResponse(BaseModel):
    accuracy: float
    correct: int
    total: int
    new_level: str
    level_changed: bool
    mastery: float
    # Cognitive load
    adaptive_mode: str = "standard"
    cognitive_strain_index: float = 0.0
    avg_response_time: float = 0.0
    # Stress detection
    stress_detected: bool = False
    recommended_action: Optional[str] = None


# --- Material upload ---
class MaterialUploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks: int
    message: str


class MaterialGenerateRequest(BaseModel):
    session_id: str
    mode: str = "lesson"  # lesson | exercise
    question_type: str = "short"
    subject: Optional[str] = None   # standalone mode
    level: Optional[str] = None     # standalone mode


class MaterialLessonResponse(BaseModel):
    lesson: str
    source: str


class MaterialExerciseResponse(BaseModel):
    questions: list[dict]
    source: str


# --- Flashcards ---
class FlashcardRequest(BaseModel):
    session_id: Optional[str] = None
    topic: Optional[str] = None
    custom_topic: Optional[str] = None   # free-form topic (no subject needed)
    from_material: bool = False
    subject: Optional[str] = None   # standalone mode (required when no session_id and no custom_topic)
    level: Optional[str] = None     # standalone mode


class FlashcardResponse(BaseModel):
    flashcards: list[dict]  # [{front, back}]
    subject: str


# --- Podcast ---
class PodcastRequest(BaseModel):
    topic: str


class PodcastScriptEntry(BaseModel):
    speaker: str
    name: str
    text: str
    emotion: str = "neutral"
    audio_url: Optional[str] = None


class PodcastResponse(BaseModel):
    podcast_id: str
    topic: str
    script: list[dict]
    full_audio_url: Optional[str] = None
    has_audio: bool = False
    segments: int = 0


# --- Enhanced progress ---
class ProgressResponse(BaseModel):
    session_id: str
    subject: str
    level: str
    total_correct: int
    total_attempts: int
    accuracy: float
    level_history: list[str]
    mastery: float
    weaknesses: list[dict]
    recommendations: list[str]
    topic_accuracy: dict
    type_accuracy: dict
    # Cognitive metrics
    cognitive_strain_index: float = 0.0
    avg_response_time: float = 0.0
    adaptive_mode: Optional[str] = None
    # Weakness DNA
    weakness_profile: dict = {}
    # Topic suggestion
    suggested_topic: Optional[str] = None


class WeaknessTopicEntry(BaseModel):
    mastery_score: float
    error_types: list[str]
    recurring_patterns: list[str]
    last_updated: Optional[str] = None


class WeaknessProfileResponse(BaseModel):
    session_id: str
    subject: str
    weakness_profile: dict  # topic -> WeaknessTopicEntry dict


# --- Authentication ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class User(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str
    userId: str
