"""
Podcast Engine — Generates two-speaker educational podcast scripts using the
configured LLM (Gemini or Mistral/Ollama, via gemini_client) and synthesises
speech with the ElevenLabs TTS API.
"""

import os
import re
import json
import uuid
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

PODCAST_DIR = Path(__file__).parent / "podcast_audio"
PODCAST_DIR.mkdir(exist_ok=True)

# ElevenLabs voice IDs  (override via env if needed)
# Defaults: Adam (deep male) & Rachel (clear female)
VOICE_MAP = {
    "host": os.getenv("ELEVENLABS_HOST_VOICE", "pNInz6obpgDQGcFmaJgB"),   # Adam
    "guest": os.getenv("ELEVENLABS_GUEST_VOICE", "21m00Tcm4TlvDq8ikWAM"),  # Rachel
}

SPEAKER_NAMES = {
    "host": "Alex",
    "guest": "Dr. Sam",
}


def _get_api_key() -> str | None:
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    return key if key else None


# ---------------------------------------------------------------------------
# 1. Script generation  (uses gemini_client → Gemini or Ollama/Mistral)
# ---------------------------------------------------------------------------

async def generate_podcast_script(topic: str) -> list[dict]:
    """Generate a two-speaker podcast script as a list of dicts."""
    from gemini_client import generate_text

    prompt = f"""You are a podcast script writer. Generate an engaging, educational podcast script about: "{topic}"

The podcast features two speakers:
- "host" (Alex): The main host who guides the conversation, asks insightful questions, and keeps things lively.
- "guest" (Dr. Sam): A knowledgeable expert who explains concepts clearly with great analogies.

Rules:
1. Create 8-14 natural-sounding exchanges.
2. Each exchange should be 1-3 sentences.
3. Include natural reactions: "Wow, that's fascinating!", "Right, exactly!", "That's a great question!" etc.
4. Keep it conversational — not stiff or academic.
5. The script should educate the listener about {topic} in an approachable way.

IMPORTANT: Respond ONLY with a valid JSON array. Each element must have:
- "speaker": either "host" or "guest"
- "text": the dialogue text (plain text, no quotes around speakers)
- "emotion": one of "neutral", "happy", "thoughtful", "excited"

Example:
[
  {{"speaker": "host", "text": "Welcome to NeuroLearn Podcast! Today we have a truly exciting topic lined up.", "emotion": "happy"}},
  {{"speaker": "guest", "text": "Thanks for having me, Alex! I'm thrilled to talk about this.", "emotion": "happy"}}
]

Generate the full script now. Return ONLY the JSON array, no extra text:"""

    try:
        raw = await generate_text(prompt, task="podcast")
        parsed = _try_parse_json_array(raw)
        if parsed:
            normalised = []
            for entry in parsed:
                normalised.append({
                    "speaker": entry.get("speaker", "host"),
                    "name": SPEAKER_NAMES.get(entry.get("speaker", "host"), "Alex"),
                    "text": entry.get("text", ""),
                    "emotion": entry.get("emotion", "neutral"),
                })
            return normalised
    except Exception as exc:
        print(f"[Podcast] Script generation error: {exc}")

    return _fallback_script(topic)


def _try_parse_json_array(raw: str) -> list[dict] | None:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("script", "dialogue", "exchanges", "podcast", "data", "conversations"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _fallback_script(topic: str) -> list[dict]:
    return [
        {"speaker": "host", "name": "Alex", "text": f"Welcome to NeuroLearn Podcast! Today we're diving deep into {topic}.", "emotion": "happy"},
        {"speaker": "guest", "name": "Dr. Sam", "text": f"Thanks for having me, Alex! {topic} is such a fascinating area with so much to unpack.", "emotion": "happy"},
        {"speaker": "host", "name": "Alex", "text": f"Let's start at the beginning. Can you give our listeners a quick overview of what {topic} actually is?", "emotion": "neutral"},
        {"speaker": "guest", "name": "Dr. Sam", "text": f"Absolutely. At its core, {topic} is about understanding complex systems and finding elegant solutions. It's one of those fields that touches almost everything in modern technology.", "emotion": "thoughtful"},
        {"speaker": "host", "name": "Alex", "text": "That's really interesting. What got you personally excited about this field?", "emotion": "neutral"},
        {"speaker": "guest", "name": "Dr. Sam", "text": "Great question! I think it was the moment I realized how much impact it has on everyday life. Once you start seeing it, you can't unsee it.", "emotion": "excited"},
        {"speaker": "host", "name": "Alex", "text": "Wow, I love that perspective. What would you say to someone who's just getting started?", "emotion": "happy"},
        {"speaker": "guest", "name": "Dr. Sam", "text": "Start with the fundamentals, be curious, and don't be afraid to experiment. The best way to learn is by doing.", "emotion": "thoughtful"},
        {"speaker": "host", "name": "Alex", "text": "Fantastic advice. Thanks so much for joining us today!", "emotion": "happy"},
        {"speaker": "guest", "name": "Dr. Sam", "text": "My pleasure! This was a lot of fun.", "emotion": "happy"},
    ]


# ---------------------------------------------------------------------------
# 2. TTS via ElevenLabs API
# ---------------------------------------------------------------------------

async def generate_tts_segment(text: str, voice_id: str) -> bytes | None:
    """
    Call ElevenLabs text-to-speech API.  Returns MP3 bytes or None on failure.
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    # Clean text for TTS
    clean = re.sub(r"[*_#`]", "", text).strip()
    if not clean:
        return None

    url = f"{ELEVENLABS_BASE}/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": clean,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.35,
            "use_speaker_boost": True,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                return resp.content
            else:
                print(f"[TTS] ElevenLabs error {resp.status_code}: {resp.text[:200]}")
                return None
    except Exception as exc:
        print(f"[TTS] Error: {exc}")
        return None


# ---------------------------------------------------------------------------
# 3. Audio utilities
# ---------------------------------------------------------------------------

def _concat_mp3s(segments: list[bytes]) -> bytes:
    """Concatenate MP3 byte-strings (binary join works for CBR mp3)."""
    return b"".join(segments)


# ---------------------------------------------------------------------------
# 4. Top-level pipeline
# ---------------------------------------------------------------------------

async def create_podcast(topic: str) -> dict:
    """
    End-to-end podcast creation:
    1. Generate script with the configured LLM (Gemini / Mistral)
    2. Synthesise speech per segment with ElevenLabs
    3. Concatenate into a full episode MP3
    """
    podcast_id = uuid.uuid4().hex[:12]

    has_tts = _get_api_key() is not None
    if not has_tts:
        print("[Podcast] ELEVENLABS_API_KEY not set — text-only mode")

    # ── Step 1: script ──
    print(f"[Podcast] Generating script for topic: {topic}")
    script = await generate_podcast_script(topic)

    # ── Step 2: TTS per segment ──
    audio_parts: list[bytes] = []

    for idx, entry in enumerate(script):
        speaker = entry.get("speaker", "host")
        text = entry.get("text", "")
        voice_id = VOICE_MAP.get(speaker, VOICE_MAP["host"])

        print(f"[Podcast] TTS {idx + 1}/{len(script)}  [{speaker}]  {text[:50]}…")

        if has_tts:
            mp3 = await generate_tts_segment(text, voice_id)
        else:
            mp3 = None

        if mp3:
            seg_name = f"{podcast_id}_seg_{idx:02d}.mp3"
            seg_path = PODCAST_DIR / seg_name
            with open(seg_path, "wb") as f:
                f.write(mp3)
            entry["audio_url"] = f"/api/podcast-audio/{seg_name}"
            audio_parts.append(mp3)
        else:
            entry["audio_url"] = None

        if "name" not in entry:
            entry["name"] = SPEAKER_NAMES.get(speaker, speaker.title())

    # ── Step 3: full episode ──
    full_audio_url = None
    if audio_parts:
        full_name = f"{podcast_id}_full.mp3"
        full_path = PODCAST_DIR / full_name
        with open(full_path, "wb") as f:
            f.write(_concat_mp3s(audio_parts))
        full_audio_url = f"/api/podcast-audio/{full_name}"

    has_audio = full_audio_url is not None
    print(f"[Podcast] Done — {len(script)} segments, audio={'yes' if has_audio else 'no'}")

    return {
        "podcast_id": podcast_id,
        "topic": topic,
        "script": script,
        "full_audio_url": full_audio_url,
        "has_audio": has_audio,
        "segments": len(script),
    }
