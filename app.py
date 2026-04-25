import base64
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


SENSORY_PROMPT = """You are a poetic guide for blind and low-vision users experiencing nature. When given a flower image, identify the species and describe it using touch, scent, sound, and emotion — never rely on color or visual language alone. Write exactly 3 sentences. First sentence: name the species and its most distinctive tactile quality. Second sentence: describe what holding or smelling it feels like. Third sentence: one line about where it grows and what it means — culturally, seasonally, or emotionally. Warm, unhurried tone. No clinical language. No bullet points."""

MUSIC_PROMPT = """You are translating a flower into a 20-second melodic phrase that lets a blind listener perceive its visual qualities through sound.

Analyze the flower in the image and return ONLY valid JSON (no markdown, no explanation outside the JSON):

{
  "flower_name": "best guess of species",
  "dominant_colors": ["#hex", "#hex"],
  "shape_descriptor": "round | tall | drooping | clustered | spiky | flat",
  "petal_count_estimate": 12,
  "texture": "waxy | velvety | papery | fuzzy",
  "symmetry": "radial | bilateral | asymmetric",
  "key": "e.g. C major or F# minor",
  "tempo_bpm": 88,
  "voice": "sine | triangle | am | fm | pluck",
  "melody": [
    {"pitch": "C4", "duration": 0.5, "velocity": 0.7}
  ],
  "encoding_explanation": "one sentence explaining how visual maps to sound"
}

ENCODING RULES (follow strictly):
- Pitch contour traces the flower's silhouette: tall flower = ascending line, drooping = descending, round = arc up then down, spiky = jagged jumps
- Number of notes = petal_count_estimate, capped between 5 and 16
- Voice mapping: waxy→pluck, velvety→sine, papery→triangle, fuzzy→am
- Tempo: drooping/melancholy 55-75, round/serene 75-95, bold/joyful 100-130
- Color saturation maps to velocity (vivid = louder, pale = softer)
- Symmetry: radial = repeating motif, bilateral = call-and-response, asymmetric = through-composed
- Pitches must stay within C4 to C6
- Use only natural notes for major keys, follow harmonic minor for minor keys

Make the melody feel genuinely musical, not random. Each flower should sound clearly different from another."""

RESULTS_CACHE: dict[str, dict[str, Any]] = {}
FLOWER_CACHE: dict[str, dict[str, Any]] = {}
PICO_REGISTERED_URL: str = ""


@dataclass
class CaptureResult:
    image_b64: str
    mime_type: str
    frame: Any


def capture_from_image_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> CaptureResult:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError("Uploaded image could not be decoded.")

    encoded_ok, jpeg_buffer = cv2.imencode(".jpg", frame)
    if not encoded_ok:
        raise RuntimeError("Failed to re-encode uploaded image.")

    image_b64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")
    return CaptureResult(image_b64=image_b64, mime_type=mime_type, frame=frame)


def capture_frame() -> CaptureResult:
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open USB camera index {camera_index}.")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Camera opened but no frame was captured.")

    encoded_ok, jpeg_buffer = cv2.imencode(".jpg", frame)
    if not encoded_ok:
        raise RuntimeError("Failed to encode captured frame as JPEG.")

    image_b64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")
    return CaptureResult(image_b64=image_b64, mime_type="image/jpeg", frame=frame)


def _extract_json(payload: str) -> dict[str, Any]:
    payload = payload.strip()
    if payload.startswith("{"):
        return json.loads(payload)

    match = re.search(r"\{.*\}", payload, re.DOTALL)
    if not match:
        raise ValueError("Model response did not contain JSON.")
    return json.loads(match.group(0))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _duration_to_tone(duration_seconds: float) -> str:
    if duration_seconds <= 0.16:
        return "16n"
    if duration_seconds <= 0.34:
        return "8n"
    if duration_seconds <= 0.72:
        return "4n"
    if duration_seconds <= 1.35:
        return "2n"
    return "1n"


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _extract_hex_colors(raw_colors: Any) -> list[str]:
    if isinstance(raw_colors, list):
        return [str(c).strip() for c in raw_colors if isinstance(c, str) and c.strip().startswith("#")]
    if isinstance(raw_colors, str) and raw_colors.strip().startswith("#"):
        return [raw_colors.strip()]
    return []


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        return None
    try:
        return int(cleaned[0:2], 16), int(cleaned[2:4], 16), int(cleaned[4:6], 16)
    except ValueError:
        return None


def _avg_color_saturation(colors: list[str]) -> float:
    if not colors:
        return 0.5
    sats: list[float] = []
    for color in colors:
        rgb = _hex_to_rgb(color)
        if not rgb:
            continue
        r, g, b = rgb
        maxc = max(r, g, b) / 255.0
        minc = min(r, g, b) / 255.0
        if maxc == 0:
            sats.append(0.0)
        else:
            sats.append((maxc - minc) / maxc)
    if not sats:
        return 0.5
    return sum(sats) / len(sats)


def _midi_to_note_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note = names[midi % 12]
    octave = (midi // 12) - 1
    return f"{note}{octave}"


def _build_scale_pitches(key: str, low_midi: int = 60, high_midi: int = 84) -> list[int]:
    key_lower = str(key or "C major").strip().lower()
    root_note = key_lower.split()[0].upper()
    root_map = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    root = root_map.get(root_note, 0)
    is_minor = "minor" in key_lower
    intervals = [0, 2, 3, 5, 7, 8, 11] if is_minor else [0, 2, 4, 5, 7, 9, 11]

    pitches: list[int] = []
    for midi in range(low_midi, high_midi + 1):
        if (midi - root) % 12 in intervals:
            pitches.append(midi)
    return pitches or [60, 62, 64, 67, 69, 72]


def _contour_indexes(shape: str, note_count: int, scale_len: int) -> list[int]:
    max_i = max(0, scale_len - 1)
    mid = max_i // 2
    s = shape.lower()
    indexes: list[int] = []

    if "tall" in s:
        for i in range(note_count):
            indexes.append(_clamp_int((i * max_i) // max(1, note_count - 1), 0, max_i))
        return indexes

    if "drooping" in s:
        for i in range(note_count):
            asc = (i * max_i) // max(1, note_count - 1)
            indexes.append(_clamp_int(max_i - asc, 0, max_i))
        return indexes

    if "spiky" in s:
        pattern = [mid, max_i, max(0, mid - 2), max_i - 1, max(0, mid - 3), max_i]
        return [pattern[i % len(pattern)] for i in range(note_count)]

    if "flat" in s:
        return [_clamp_int(mid + (-1 if i % 3 == 0 else 0), 0, max_i) for i in range(note_count)]

    if "clustered" in s:
        pattern = [mid, mid + 1, mid, mid + 2, mid + 1]
        return [_clamp_int(pattern[i % len(pattern)], 0, max_i) for i in range(note_count)]

    # default round/arc behavior
    for i in range(note_count):
        phase = i / max(1, note_count - 1)
        if phase <= 0.5:
            idx = int((phase / 0.5) * max_i)
        else:
            idx = int(((1.0 - phase) / 0.5) * max_i)
        indexes.append(_clamp_int(idx, 0, max_i))
    return indexes


def deterministic_music_from_attributes(raw: dict[str, Any]) -> dict[str, Any]:
    flower_name = str(raw.get("flower_name") or raw.get("species") or "unknown flower")
    shape = str(raw.get("shape_descriptor") or raw.get("mood") or "round").lower()
    texture = str(raw.get("texture") or raw.get("description_hint") or "papery").lower()
    symmetry = str(raw.get("symmetry") or "radial").lower()
    key = str(raw.get("key") or "C major")
    colors = _extract_hex_colors(raw.get("dominant_colors") or raw.get("dominant_color"))

    petal_count = _safe_int(raw.get("petal_count_estimate"), len(raw.get("notes", [])) or 8)
    note_count = _clamp_int(petal_count, 5, 16)

    if "drooping" in shape:
        tempo = 64
        duration_seconds = 0.7
    elif "spiky" in shape:
        tempo = 112
        duration_seconds = 0.28
    elif "round" in shape:
        tempo = 84
        duration_seconds = 0.45
    elif "clustered" in shape:
        tempo = 98
        duration_seconds = 0.32
    else:
        tempo = 90
        duration_seconds = 0.4

    voice_map = {
        "waxy": "pluck",
        "velvety": "sine",
        "papery": "triangle",
        "fuzzy": "am",
    }
    voice = voice_map.get(texture, str(raw.get("voice") or "triangle"))

    sat = _avg_color_saturation(colors)
    base_velocity = 0.45 + (sat * 0.45)

    scale = _build_scale_pitches(key)
    contour = _contour_indexes(shape, note_count, len(scale))
    melody: list[dict[str, Any]] = []

    for i, idx in enumerate(contour):
        midi = scale[_clamp_int(idx, 0, len(scale) - 1)]
        pitch = _midi_to_note_name(midi)
        velocity = max(0.35, min(0.95, base_velocity + (0.06 if i % 4 == 0 else 0.0)))
        melody.append(
            {
                "pitch": pitch,
                "duration": round(duration_seconds, 2),
                "velocity": round(velocity, 2),
            }
        )

    if symmetry == "radial" and len(melody) >= 8:
        motif = melody[:4]
        repeat_count = len(melody) // len(motif)
        melody = (motif * repeat_count) + motif[: len(melody) % len(motif)]
    elif symmetry == "bilateral" and len(melody) >= 6:
        half = len(melody) // 2
        first = melody[:half]
        second = list(reversed(first))
        melody = (first + second)[: len(melody)]

    return {
        "flower_name": flower_name,
        "dominant_colors": colors or ["#8fcf7a", "#f5f0c8"],
        "shape_descriptor": shape,
        "petal_count_estimate": note_count,
        "texture": texture,
        "symmetry": symmetry,
        "key": key,
        "tempo_bpm": tempo,
        "voice": voice,
        "melody": melody,
        "encoding_explanation": "Deterministic mapping uses shape, texture, symmetry, petal count, and color saturation to derive contour, timbre, motif, note count, and dynamics.",
    }


def _image_fingerprint(frame: Any) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    avg = float(resized.mean())
    bits = "".join("1" if int(px) >= avg else "0" for row in resized for px in row)
    hex_length = len(bits) // 4
    return f"{int(bits, 2):0{hex_length}x}"


def _normalize_music_json(raw: dict[str, Any]) -> dict[str, Any]:
    # Supports both old schema (notes/durations/beats) and S-aligned schema (melody array).
    if isinstance(raw.get("melody"), list):
        melody = raw["melody"]
        notes: list[str] = []
        durations: list[str] = []
        beats: list[int] = []
        elapsed_ms = 0.0
        for item in melody:
            if not isinstance(item, dict):
                continue
            pitch = item.get("pitch", "C4")
            dur_seconds = max(0.1, _safe_float(item.get("duration"), 0.5))
            notes.append(str(pitch))
            durations.append(_duration_to_tone(dur_seconds))
            beats.append(int(round(elapsed_ms)))
            elapsed_ms += dur_seconds * 1000.0

        return {
            "species": raw.get("flower_name", "unknown flower"),
            "key": raw.get("key", "C major"),
            "tempo": _safe_int(raw.get("tempo_bpm"), 84),
            "scale": "major" if "major" in str(raw.get("key", "")).lower() else "minor",
            "notes": notes or ["C4", "E4", "G4", "E4"],
            "durations": durations or ["4n", "4n", "4n", "4n"],
            "articulation": "legato",
            "mood": raw.get("shape_descriptor", "gentle"),
            "description_hint": raw.get("texture", "soft"),
            "dominant_color": ", ".join(raw.get("dominant_colors", [])[:2]) or "spring green",
            "beats": beats or [0, 500, 1000, 1500],
            "voice": raw.get("voice", "triangle"),
        }

    beats = raw.get("beats", [])
    if not isinstance(beats, list):
        beats = []
    return {
        "species": raw.get("species", "unknown flower"),
        "key": raw.get("key", "D major"),
        "tempo": _safe_int(raw.get("tempo"), 72),
        "scale": raw.get("scale", "major"),
        "notes": raw.get("notes", ["D4", "F#4", "A4", "D5"]),
        "durations": raw.get("durations", ["4n", "8n", "8n", "2n"]),
        "articulation": raw.get("articulation", "legato"),
        "mood": raw.get("mood", "gentle"),
        "description_hint": raw.get("description_hint", "calm"),
        "dominant_color": raw.get("dominant_color", "soft yellow"),
        "beats": beats or [0, 500, 1000, 1500],
        "voice": raw.get("voice", "triangle"),
    }


def _anthropic_client() -> Any:
    try:
        from anthropic import Anthropic  # Imported lazily so Ollama-only usage works.
    except ImportError as exc:
        raise RuntimeError(
            "Anthropic SDK not installed. Run `pip install anthropic` or switch LLM_PROVIDER=ollama."
        ) from exc

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing in environment.")
    return Anthropic(api_key=api_key)


def ask_claude_with_image(prompt: str, capture: CaptureResult) -> str:
    model = os.getenv("CLAUDE_MODEL", "claude-3-7-sonnet-latest")
    client = _anthropic_client()
    message = client.messages.create(
        model=model,
        max_tokens=900,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": capture.mime_type,
                            "data": capture.image_b64,
                        },
                    },
                ],
            }
        ],
    )

    chunks: list[str] = []
    for block in message.content:
        if getattr(block, "type", "") == "text":
            chunks.append(block.text)
    return "\n".join(chunks).strip()


def ask_ollama_with_image(prompt: str, capture: CaptureResult) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llava:7b")
    timeout_s = _safe_int(os.getenv("OLLAMA_TIMEOUT_S", "60"), 60)
    endpoint = f"{base_url}/api/chat"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [capture.image_b64]}],
        "stream": False,
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    message = data.get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned empty response content.")
    return content.strip()


def ask_model_with_image(prompt: str, capture: CaptureResult) -> str:
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "anthropic":
        return ask_claude_with_image(prompt, capture)
    if provider == "ollama":
        return ask_ollama_with_image(prompt, capture)
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{provider}'. Use 'ollama' or 'anthropic'.")


def generate_mock_music_raw() -> dict[str, Any]:
    return {
        "flower_name": "daisy",
        "dominant_colors": ["#f6e27f", "#ffffff"],
        "shape_descriptor": "round",
        "petal_count_estimate": 12,
        "texture": "papery",
        "symmetry": "radial",
        "key": "D major",
        "tempo_bpm": 84,
        "voice": "triangle",
        "melody": [
            {"pitch": "D4", "duration": 0.5, "velocity": 0.65},
            {"pitch": "F#4", "duration": 0.3, "velocity": 0.68},
            {"pitch": "A4", "duration": 0.3, "velocity": 0.7},
            {"pitch": "B4", "duration": 0.5, "velocity": 0.68},
            {"pitch": "A4", "duration": 0.5, "velocity": 0.66},
            {"pitch": "F#4", "duration": 0.3, "velocity": 0.64},
            {"pitch": "E4", "duration": 0.3, "velocity": 0.62},
            {"pitch": "D4", "duration": 0.8, "velocity": 0.6},
        ],
        "encoding_explanation": "A rounded repeating arc and gentle mid tempo reflect radial petals and a light papery texture.",
    }


def generate_mock_description() -> str:
    return (
        "This is a daisy, and its petals feel like soft paper layered around a small raised center. "
        "In your hand it feels light and breathable, with a clean meadow scent that settles your breathing. "
        "It grows in open fields and roadsides, carrying the feeling of spring simplicity and quiet resilience."
    )


def _resolve_pico_url() -> str:
    if PICO_REGISTERED_URL:
        return PICO_REGISTERED_URL
    return os.getenv("PICO_HAPTIC_URL", "").strip()


def forward_haptic(beats: list[int]) -> dict[str, Any]:
    pico_url = _resolve_pico_url()
    if not pico_url:
        return {"forwarded": False, "reason": "PICO endpoint not configured"}

    try:
        response = requests.post(pico_url, json={"beats": beats}, timeout=4)
        return {
            "forwarded": response.ok,
            "status_code": response.status_code,
            "url": pico_url,
        }
    except requests.RequestException as exc:
        return {"forwarded": False, "reason": str(exc), "url": pico_url}


def _save_capture_image(frame: Any, capture_id: str) -> str:
    captures_dir = Path(app.static_folder) / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)
    image_path = captures_dir / f"flower_{capture_id}.jpg"
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError("Could not save captured image to static folder.")
    return f"/static/captures/flower_{capture_id}.jpg"


def run_pipeline_for_capture(capture: CaptureResult, capture_id: str | None = None) -> dict[str, Any]:
    started_at = time.time()
    fingerprint = _image_fingerprint(capture.frame)
    use_mock = os.getenv("USE_MOCK", "").lower() in {"1", "true", "yes"}
    cache_hit = fingerprint in FLOWER_CACHE

    if cache_hit:
        cached_flower = FLOWER_CACHE[fingerprint]
        sensory = cached_flower["sensory_description"]
        music_raw = cached_flower["melody_json"]
    else:
        if use_mock:
            sensory = generate_mock_description()
            extracted_raw = generate_mock_music_raw()
        else:
            sensory = ask_model_with_image(SENSORY_PROMPT, capture)
            music_raw_text = ask_model_with_image(MUSIC_PROMPT, capture)
            extracted_raw = _extract_json(music_raw_text)

        music_raw = deterministic_music_from_attributes(extracted_raw)
        FLOWER_CACHE[fingerprint] = {
            "sensory_description": sensory,
            "melody_json": music_raw,
        }

    music = _normalize_music_json(music_raw)
    beats = music.get("beats", [])
    haptic_status = forward_haptic(beats if isinstance(beats, list) else [])
    elapsed_ms = int((time.time() - started_at) * 1000)

    image_url = ""
    if capture_id:
        image_url = _save_capture_image(capture.frame, capture_id)

    return {
        "capture_id": capture_id,
        "flower_fingerprint": fingerprint,
        "cache_hit": cache_hit,
        "image_url": image_url,
        "sensory_description": sensory,
        "melody_json": music_raw,  # S-aligned raw schema
        "music": music,  # frontend-friendly normalized schema
        "haptic": haptic_status,
        "latency_ms": elapsed_ms,
    }


def run_pipeline(capture_id: str | None = None) -> dict[str, Any]:
    capture = capture_frame()
    return run_pipeline_for_capture(capture=capture, capture_id=capture_id)


def _camera_health() -> dict[str, Any]:
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"ok": False, "camera_index": camera_index, "reason": "unable to open camera"}

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"ok": False, "camera_index": camera_index, "reason": "camera opened but no frame"}
    return {"ok": True, "camera_index": camera_index}


def _provider_health() -> dict[str, Any]:
    use_mock = os.getenv("USE_MOCK", "").lower() in {"1", "true", "yes"}
    if use_mock:
        return {"ok": True, "provider": "mock", "reason": "USE_MOCK is enabled"}

    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "llava:7b")
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            if not response.ok:
                return {"ok": False, "provider": "ollama", "reason": f"status {response.status_code}"}
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            has_model = any(name == model or name.startswith(f"{model}:") for name in models)
            return {
                "ok": has_model,
                "provider": "ollama",
                "base_url": base_url,
                "model": model,
                "reason": "model available" if has_model else "model not found, run `ollama pull`",
            }
        except requests.RequestException as exc:
            return {"ok": False, "provider": "ollama", "reason": str(exc)}

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        return {
            "ok": bool(api_key),
            "provider": "anthropic",
            "reason": "configured" if api_key else "missing ANTHROPIC_API_KEY",
        }

    return {"ok": False, "provider": provider, "reason": "unsupported provider"}


def _haptics_health() -> dict[str, Any]:
    pico_url = _resolve_pico_url()
    if not pico_url:
        return {"ok": False, "configured": False, "reason": "PICO endpoint not configured"}
    return {"ok": True, "configured": True, "url": pico_url}


def _camera_required() -> bool:
    return os.getenv("REQUIRE_USB_CAMERA", "").lower() in {"1", "true", "yes"}


@app.get("/")
def index() -> Any:
    return render_template("index.html")


@app.get("/health")
def health() -> Any:
    provider = _provider_health()
    camera = _camera_health()
    haptics = _haptics_health()
    camera_required = _camera_required()
    overall_ok = bool(provider.get("ok")) and (bool(camera.get("ok")) or not camera_required)

    status_code = 200 if overall_ok else 503
    return (
        jsonify(
            {
                "ok": overall_ok,
                "camera_required": camera_required,
                "provider": provider,
                "camera": camera,
                "haptics": haptics,
                "cache_size": len(RESULTS_CACHE),
                "flower_cache_size": len(FLOWER_CACHE),
            }
        ),
        status_code,
    )


@app.post("/analyze-photo")
def analyze_photo() -> Any:
    capture_id = request.args.get("id", "").strip() or uuid.uuid4().hex[:8]
    try:
        uploaded = request.files.get("photo")
        payload = request.get_json(silent=True) or {}

        capture: CaptureResult
        if uploaded and uploaded.readable():
            image_bytes = uploaded.read()
            mime_type = uploaded.mimetype or "image/jpeg"
            capture = capture_from_image_bytes(image_bytes=image_bytes, mime_type=mime_type)
        elif payload.get("image_base64"):
            b64 = str(payload.get("image_base64"))
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
            mime_type = str(payload.get("mime_type") or "image/jpeg")
            capture = capture_from_image_bytes(image_bytes=image_bytes, mime_type=mime_type)
        else:
            return jsonify({"error": "No image provided. Send multipart field `photo` or JSON `image_base64`."}), 400

        result = run_pipeline_for_capture(capture=capture, capture_id=capture_id)
        RESULTS_CACHE[capture_id] = result
        return jsonify({"id": capture_id, "result": result, "melody_json": result["melody_json"], "music": result["music"]})
    except Exception as exc:
        return jsonify({"error": str(exc), "id": capture_id}), 500


@app.post("/capture")
def capture_endpoint() -> Any:
    capture_id = request.args.get("id", "").strip() or uuid.uuid4().hex[:8]
    try:
        result = run_pipeline(capture_id=capture_id)
        RESULTS_CACHE[capture_id] = result
        return jsonify({"id": capture_id, "image_url": result["image_url"], "melody_json": result["melody_json"], "result": result})
    except Exception as exc:
        return jsonify({"error": str(exc), "id": capture_id}), 500


@app.get("/result/<capture_id>")
def get_result(capture_id: str) -> Any:
    cached = RESULTS_CACHE.get(capture_id)
    if not cached:
        return jsonify({"error": "Unknown capture id"}), 404
    return jsonify(cached)


@app.post("/register")
def register_pico() -> Any:
    global PICO_REGISTERED_URL

    payload = request.get_json(silent=True) or {}
    ip = str(payload.get("ip", "")).strip()
    path = str(payload.get("path", "/haptic")).strip() or "/haptic"
    port = _safe_int(payload.get("port", 80), 80)

    if not ip:
        return jsonify({"error": "Missing `ip` in payload"}), 400

    if not path.startswith("/"):
        path = f"/{path}"

    PICO_REGISTERED_URL = f"http://{ip}:{port}{path}"
    return jsonify({"ok": True, "pico_url": PICO_REGISTERED_URL})


@app.post("/beats/<capture_id>")
def beats_for_capture(capture_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    beats = payload.get("beats") or payload.get("beats_ms")

    if not isinstance(beats, list):
        cached = RESULTS_CACHE.get(capture_id, {})
        beats = cached.get("music", {}).get("beats", [])

    if not isinstance(beats, list):
        beats = []

    status = forward_haptic(beats)
    return jsonify({"id": capture_id, "beats_count": len(beats), "haptic": status})


@app.post("/trigger")
def trigger() -> Any:
    try:
        capture_id = uuid.uuid4().hex[:8]
        result = run_pipeline(capture_id=capture_id)
        RESULTS_CACHE[capture_id] = result
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/button-pressed")
def button_pressed() -> Any:
    # Pico W calls this endpoint after the physical button is pressed.
    return trigger()


@app.post("/haptic")
def haptic_proxy() -> Any:
    payload = request.get_json(silent=True) or {}
    beats = payload.get("beats") or payload.get("beats_ms") or []
    status = forward_haptic(beats if isinstance(beats, list) else [])
    return jsonify(status)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
