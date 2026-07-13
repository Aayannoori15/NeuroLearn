"""
Multi-model AI client for NeuroLearn.

Routing strategy:
  - Complex tasks (lesson, exercise, diagnostic) → gemini-2.5-flash (primary)
  - Lighter tasks (flashcard, podcast)           → gemini-2.0-flash-lite (secondary)

Fallback chain (automatic, no config needed):
  1. Primary Gemini model for the task
  2. Secondary Gemini model (if different from primary)
  3. Ollama / Mistral (local)

The public API is unchanged: generate_text(prompt) / generate_json(prompt).
All functions accept an optional task= keyword to enable routing.
"""

import os
import json
import re
import asyncio
import httpx

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

GEMINI_MODELS: dict[str, str] = {
    "primary":   os.getenv("GEMINI_PRIMARY_MODEL",   "gemini-2.5-flash"),
    "secondary": os.getenv("GEMINI_SECONDARY_MODEL", "gemini-2.0-flash-lite"),
}

OLLAMA_BASE = "http://localhost:11434"

# Task → model tier
# "primary"   → deeper reasoning (lessons, exercises, diagnostics)
# "secondary" → lighter / faster tasks (flashcards, podcast scripts)
TASK_MODEL_MAP: dict[str, str] = {
    "lesson":     "primary",
    "exercise":   "primary",
    "diagnostic": "primary",
    "flashcard":  "secondary",
    "podcast":    "secondary",
    "default":    "primary",
}

# ---------------------------------------------------------------------------
# Gemini client (lazy, shared singleton)
# ---------------------------------------------------------------------------

_gemini_client = None


def _get_gemini_client():
    """Lazy-init the Gemini SDK client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment / .env")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Retry-delay parser (for 429 responses)
# ---------------------------------------------------------------------------

def _parse_retry_delay(exc: Exception) -> float | None:
    """Try to extract the retryDelay seconds from a 429 error message."""
    msg = str(exc)
    m = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"retryDelay['\"]:\s*['\"]?([\d.]+)s", msg)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Low-level provider calls
# ---------------------------------------------------------------------------

async def _gemini_text(prompt: str, model: str) -> str:
    """Call Gemini for a text response."""
    client = _get_gemini_client()
    response = await client.aio.models.generate_content(model=model, contents=prompt)
    return response.text


async def _gemini_json_raw(prompt: str, model: str) -> str:
    """Call Gemini with JSON response MIME type and return raw text."""
    from google.genai import types
    client = _get_gemini_client()
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return response.text


async def _ollama_generate(prompt: str, json_mode: bool = False) -> str:
    """Call local Ollama API and return the response text."""
    model = os.getenv("OLLAMA_MODEL", "mistral")
    payload: dict = {"model": model, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")


# ---------------------------------------------------------------------------
# Provider chain builder
# ---------------------------------------------------------------------------

def _build_provider_chain(task: str) -> list[dict]:
    """
    Build an ordered list of providers to try for the given task.

    Each entry:
        {"kind": "gemini", "model": "<model-name>"}
      | {"kind": "ollama",  "model": None}

    Chain order:
        1. Task-appropriate Gemini model  (if GEMINI_API_KEY is set)
        2. Secondary Gemini model          (if different from primary)
        3. Ollama / Mistral               (always last)
    """
    tier      = TASK_MODEL_MAP.get(task, "primary")
    primary   = GEMINI_MODELS[tier]
    secondary = GEMINI_MODELS["secondary"]

    chain: list[dict] = []

    if _has_gemini_key():
        chain.append({"kind": "gemini", "model": primary})
        # Only add secondary if it's a different model
        if primary != secondary:
            chain.append({"kind": "gemini", "model": secondary})

    # Ollama is always the final fallback
    chain.append({"kind": "ollama", "model": None})

    return chain


def _provider_label(provider: dict) -> str:
    if provider["kind"] == "gemini":
        return provider["model"]
    return f"Ollama/{os.getenv('OLLAMA_MODEL', 'mistral')}"


# ---------------------------------------------------------------------------
# JSON parsing utility
# ---------------------------------------------------------------------------

def _parse_json_text(raw: str) -> list[dict] | None:
    """Try hard to extract a JSON array from raw LLM text. Returns None on failure."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("data", "questions", "items", "results", "quiz", "flashcards"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            if "question" in parsed:
                return [parsed]
        return [parsed] if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_text(prompt: str, task: str = "default") -> str:
    """
    Send a prompt and return a text response.

    Tries each provider in the fallback chain in order.
    On any failure, automatically moves to the next provider.

    Args:
        prompt: The full prompt string.
        task:   One of "lesson", "exercise", "diagnostic", "flashcard",
                "podcast", or "default". Controls which model is used first.
    """
    chain = _build_provider_chain(task)

    for idx, provider in enumerate(chain):
        label = _provider_label(provider)
        try:
            if provider["kind"] == "gemini":
                result = await _gemini_text(prompt, provider["model"])
            else:
                result = await _ollama_generate(prompt)

            if idx > 0:
                print(f"[AI] Succeeded with fallback provider: {label}  (task={task})")
            else:
                print(f"[AI] generate_text: {label}  (task={task})")
            return result

        except Exception as exc:
            print(f"[AI] {label} failed (task={task}): {exc}")
            if idx < len(chain) - 1:
                print(f"[AI] Falling back to next provider...")

    return "[Error] All AI providers failed. Please try again."


async def generate_json(prompt: str, task: str = "default", retries: int = 3) -> list[dict]:
    """
    Generate a JSON array response with task-based routing and provider fallback.

    For each provider in the chain, retries up to `retries` times on parse
    failures or rate-limit errors before moving to the next provider.

    Args:
        prompt:  The full prompt string.
        task:    Task type for model routing (see TASK_MODEL_MAP).
        retries: Max per-provider retry attempts on failure.
    """
    chain = _build_provider_chain(task)

    for provider_idx, provider in enumerate(chain):
        label    = _provider_label(provider)
        last_raw = ""

        for attempt in range(1 + retries):
            try:
                if provider["kind"] == "gemini":
                    last_raw = await _gemini_json_raw(prompt, provider["model"])
                else:
                    json_prompt = (
                        prompt
                        + "\n\nIMPORTANT: Respond ONLY with a valid JSON array. "
                        "No markdown, no explanation, just the JSON array."
                    )
                    last_raw = await _ollama_generate(json_prompt, json_mode=True)

            except Exception as exc:
                print(f"[AI] {label} error (attempt {attempt + 1}, task={task}): {exc}")
                if attempt < retries:
                    delay = _parse_retry_delay(exc)
                    wait  = min(delay + 2, 60) if delay else 2 ** (attempt + 1)
                    print(f"[AI] Retrying {label} in {wait:.0f}s...")
                    await asyncio.sleep(wait)
                    continue
                # Exhausted per-provider retries → move to next provider
                print(f"[AI] {label} exhausted retries — trying next provider...")
                break  # inner loop → outer loop advances provider

            parsed = _parse_json_text(last_raw)
            if parsed is not None:
                if provider_idx > 0:
                    print(f"[AI] JSON succeeded with fallback provider: {label}  (task={task})")
                else:
                    print(f"[AI] generate_json: {label}  (task={task})")
                return parsed

            if attempt < retries:
                print(f"[AI] {label} malformed JSON, retrying ({attempt + 1}/{retries}, task={task})...")
                await asyncio.sleep(1)

    print(f"[AI] All providers failed for generate_json (task={task}). Last raw: {last_raw[:200]}")
    return [{"question": "AI request failed. Please try again.", "answer": "N/A"}]
