import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
GEMINI_MODEL = "gemini-3.5-transcribe"
GEMINI_API = "https://generativelanguage.googleapis.com"
DEFAULT_LANGUAGE = os.environ.get("LANGUAGE", "ml")
DEFAULT_DECODER = os.environ.get("DECODER", "rnnt")
DEFAULT_ENGINE = "indic"
ENGINES = {"indic", "google"}
GEMINI_LANGUAGES = {"ml": "ml-IN"}
GOOGLE_MAX_SECONDS = 60 * 60
MAX_BYTES = 100 * 1024 * 1024
SAMPLE_RATE = 16000
CHUNK_SECONDS = 60
MIN_CHUNK_SAMPLES = SAMPLE_RATE // 2
TEMPLATE = Path(__file__).parent / "templates" / "index.html"
EVALUATE_TEMPLATE = Path(__file__).parent / "templates" / "evaluate.html"
LANGUAGES = {
    "as",
    "bn",
    "brx",
    "doi",
    "gu",
    "hi",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga"}

state = {"model": None, "error": None, "loading": True}
infer_lock = threading.Lock()


def huggingface_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token or None


def gemini_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def friendly_load_error(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if any(marker in lowered for marker in ("401", "403", "gated", "authorized", "token", "authenticate")):
        return (
            "Hugging Face blocked the model download. Open "
            "https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual, "
            "accept the conditions, create a read token, and start the app with HF_TOKEN set."
        )
    return f"Could not load the model: {message}"


def load_model() -> None:
    try:
        import torch
        from transformers import AutoModel

        torch.set_num_threads(max(1, os.cpu_count() or 1))
        loaded = AutoModel.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            token=huggingface_token(),
        )
        loaded.eval()
        state["model"] = loaded
        state["error"] = None
    except Exception as exc:
        state["model"] = None
        state["error"] = friendly_load_error(exc)
    finally:
        state["loading"] = False


def start_model_load() -> None:
    if os.environ.get("SKIP_MODEL_LOAD") == "1":
        state["loading"] = False
        return
    threading.Thread(target=load_model, name="model-load", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_model_load()
    yield


app = FastAPI(title="Malayalam transcription", lifespan=lifespan)


def as_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return as_text(value[0]) if value else ""
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    return str(value)


def export_16k_wav(src: Path) -> Path:
    dst = src.with_suffix(".16k.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
            check=True,
            capture_output=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=400, detail="Audio conversion timed out.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = stderr[-1] if stderr else "Could not read that audio file."
        raise HTTPException(status_code=400, detail=detail) from exc
    return dst


def to_16k_mono(src: Path):
    import soundfile as sf
    import torch

    dst = export_16k_wav(src)
    audio, sample_rate = sf.read(dst, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T)
    if sample_rate != 16000:
        new_length = max(1, int(round(wav.shape[-1] * 16000 / sample_rate)))
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(0),
            size=new_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
    return torch.mean(wav, dim=0, keepdim=True)


def chunk_ranges(total_samples: int, sample_rate: int = SAMPLE_RATE, chunk_seconds: int = CHUNK_SECONDS) -> list[tuple[int, int]]:
    if total_samples <= 0:
        return []
    chunk = sample_rate * chunk_seconds
    ranges = []
    start = 0
    while start < total_samples:
        end = min(start + chunk, total_samples)
        ranges.append((start, end))
        start = end
    if len(ranges) > 1:
        ranges = [(start, end) for start, end in ranges if end - start >= MIN_CHUNK_SAMPLES]
    return ranges


def transcribe_wav(wav, language: str, decoder: str) -> str:
    import torch

    model = state["model"]
    parts = []
    with infer_lock:
        with torch.inference_mode():
            for start, end in chunk_ranges(wav.shape[-1]):
                text = as_text(model(wav[:, start:end], language, decoder)).strip()
                if text:
                    parts.append(text)
    return " ".join(parts)


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\u200c", "").replace("\u200d", "").split())


def levenshtein(left: list, right: list) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, item in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            cost = 0 if item == other else 1
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, by_word: bool) -> float | None:
    ref_text = normalize_text(reference)
    hyp_text = normalize_text(hypothesis)
    if not ref_text:
        return None
    ref_units = ref_text.split() if by_word else list(ref_text)
    hyp_units = hyp_text.split() if by_word else list(hyp_text)
    if not ref_units:
        return None
    return round(levenshtein(ref_units, hyp_units) / len(ref_units), 4)


def _align(left: list, right: list) -> list[tuple[str, str, str]]:
    rows = len(left) + 1
    cols = len(right) + 1
    costs = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
    for j in range(1, cols):
        costs[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            change = 0 if left[i - 1] == right[j - 1] else 1
            costs[i][j] = min(
                costs[i - 1][j] + 1,
                costs[i][j - 1] + 1,
                costs[i - 1][j - 1] + change,
            )

    ops = []
    i, j = len(left), len(right)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] and left[i - 1] == right[j - 1]:
            ops.append(("equal", left[i - 1], right[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] + 1:
            ops.append(("replace", left[i - 1], right[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and costs[i][j] == costs[i - 1][j] + 1:
            ops.append(("delete", left[i - 1], ""))
            i -= 1
        else:
            ops.append(("insert", "", right[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _append_span(spans: list[dict], kind: str, text: str) -> None:
    if not text:
        return
    if spans and spans[-1]["kind"] == kind:
        spans[-1]["text"] += text
        return
    spans.append({"kind": kind, "text": text})


def diff_spans(reference: str, hypothesis: str) -> dict | None:
    ref_text = normalize_text(reference)
    if not ref_text:
        return None
    hyp_text = normalize_text(hypothesis)
    reference_spans: list[dict] = []
    hypothesis_spans: list[dict] = []
    for op, ref_char, hyp_char in _align(list(ref_text), list(hyp_text)):
        if op == "equal":
            _append_span(reference_spans, "equal", ref_char)
            _append_span(hypothesis_spans, "equal", hyp_char)
        elif op == "replace":
            _append_span(reference_spans, "change", ref_char)
            _append_span(hypothesis_spans, "change", hyp_char)
        elif op == "delete":
            _append_span(reference_spans, "change", ref_char)
        else:
            _append_span(hypothesis_spans, "change", hyp_char)
    return {"reference": reference_spans, "hypothesis": hypothesis_spans}


def gemini_failure(status: int, body: bytes) -> HTTPException:
    message = ""
    try:
        message = json.loads(body.decode()).get("error", {}).get("message", "")
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        message = ""
    if status in {401, 403}:
        return HTTPException(
            status_code=503,
            detail="Google rejected GEMINI_API_KEY. Create a key in Google AI Studio and put it in .env.",
        )
    if status == 429:
        return HTTPException(
            status_code=429,
            detail="The Google free limit for this model is used up. Wait until it resets, or check the quota in AI Studio.",
        )
    if message:
        return HTTPException(status_code=502, detail=f"Google transcription failed: {message[:300]}")
    return HTTPException(status_code=502, detail="Google transcription failed.")


def gemini_transcript(payload: dict) -> str:
    parts = []
    for candidate in payload.get("candidates") or []:
        content = candidate.get("content") or {}
        for part in content.get("parts") or []:
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return " ".join(parts)


def _gemini_call(url: str, data: bytes | None, headers: dict, method: str, timeout: int) -> dict:
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    request.add_header("x-goog-api-key", gemini_api_key())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise gemini_failure(exc.code, exc.read()) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail="Could not reach Google.") from exc
    if not raw:
        return {}
    return json.loads(raw.decode())


def _gemini_upload(wav_path: Path) -> dict:
    audio = wav_path.read_bytes()
    start = urllib.request.Request(
        f"{GEMINI_API}/upload/v1beta/files",
        data=json.dumps({"file": {"display_name": wav_path.name}}).encode(),
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(audio)),
            "X-Goog-Upload-Header-Content-Type": "audio/wav",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start.add_header("x-goog-api-key", gemini_api_key())
    try:
        with urllib.request.urlopen(start, timeout=60) as response:
            upload_url = response.headers.get("x-goog-upload-url")
    except urllib.error.HTTPError as exc:
        raise gemini_failure(exc.code, exc.read()) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail="Could not reach Google.") from exc
    if not upload_url:
        raise HTTPException(status_code=502, detail="Google did not accept the audio.")
    payload = _gemini_call(
        upload_url.strip(),
        audio,
        {
            "Content-Length": str(len(audio)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        "POST",
        600,
    )
    uploaded = payload.get("file") or payload
    if not uploaded.get("uri") or not uploaded.get("name"):
        raise HTTPException(status_code=502, detail="Google did not accept the audio.")
    return uploaded


def _wait_for_gemini_file(uploaded: dict) -> None:
    name = uploaded["name"]
    state = uploaded.get("state") or "ACTIVE"
    for _ in range(30):
        if state == "ACTIVE":
            return
        if state == "FAILED":
            raise HTTPException(status_code=502, detail="Google could not read that audio.")
        time.sleep(1)
        info = _gemini_call(f"{GEMINI_API}/v1beta/{name}", None, {}, "GET", 30)
        state = (info.get("state") or info.get("file", {}).get("state") or "ACTIVE")
    raise HTTPException(status_code=504, detail="Google took too long to accept the audio.")


def _gemini_delete(name: str) -> None:
    if not name:
        return
    try:
        _gemini_call(f"{GEMINI_API}/v1beta/{name}", None, {}, "DELETE", 30)
    except HTTPException:
        return


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (wave.Error, EOFError) as exc:
        raise HTTPException(status_code=400, detail="Could not read that audio file.") from exc
    if rate <= 0:
        raise HTTPException(status_code=400, detail="Could not read that audio file.")
    return frames / rate


def transcribe_gemini(wav_path: Path, language: str) -> str:
    if wav_seconds(wav_path) > GOOGLE_MAX_SECONDS:
        raise HTTPException(status_code=400, detail="Google accepts up to one hour of audio.")
    uploaded = _gemini_upload(wav_path)
    try:
        _wait_for_gemini_file(uploaded)
        payload = _gemini_call(
            f"{GEMINI_API}/v1beta/models/{GEMINI_MODEL}:generateContent",
            json.dumps(
                {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "fileData": {
                                        "mimeType": uploaded.get("mimeType") or "audio/wav",
                                        "fileUri": uploaded["uri"],
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "audioTranscriptionConfig": {"languageCodes": [GEMINI_LANGUAGES[language]]}
                    },
                }
            ).encode(),
            {"Content-Type": "application/json"},
            "POST",
            600,
        )
    finally:
        _gemini_delete(uploaded.get("name", ""))
    return gemini_transcript(payload)


def transcription_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) > 300:
        message = message[:300] + "…"
    return f"Transcription failed: {message}"


@app.get("/health")
def health() -> dict:
    google = "ready" if gemini_api_key() else "missing"
    if state["error"]:
        return {"status": "error", "detail": state["error"], "model": MODEL_ID, "google": google}
    if state["loading"] or state["model"] is None:
        return {"status": "loading", "model": MODEL_ID, "google": google}
    return {"status": "ready", "model": MODEL_ID, "language": DEFAULT_LANGUAGE, "google": google}


def require_engine(engine: str, decoder: str, language: str) -> None:
    if engine not in ENGINES:
        raise HTTPException(status_code=400, detail="Model must be indic or google.")
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language id.")
    if engine == "google":
        if language not in GEMINI_LANGUAGES:
            raise HTTPException(status_code=400, detail="The Google model in this app transcribes Malayalam.")
        if not gemini_api_key():
            raise HTTPException(status_code=503, detail="Add GEMINI_API_KEY to .env to use the Google model.")
        return
    if decoder not in {"ctc", "rnnt"}:
        raise HTTPException(status_code=400, detail="Decoder must be ctc or rnnt.")
    if state["error"]:
        raise HTTPException(status_code=503, detail=state["error"])
    if state["loading"] or state["model"] is None:
        raise HTTPException(status_code=503, detail="The model is still downloading.")


def transcribe_bytes(payload: bytes, filename: str, language: str, decoder: str, engine: str) -> str:
    if not payload:
        raise HTTPException(status_code=400, detail="The audio file is empty.")
    if len(payload) > MAX_BYTES:
        limit_mb = MAX_BYTES // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"The file is larger than {limit_mb} MB.")

    suffix = Path(filename or "audio.wav").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".audio"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{suffix}"
        src.write_bytes(payload)
        try:
            if engine == "google":
                return transcribe_gemini(export_16k_wav(src), language)
            return transcribe_wav(to_16k_mono(src), language, decoder)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=transcription_error(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(TEMPLATE)


@app.get("/evaluate")
def evaluate_page() -> FileResponse:
    return FileResponse(EVALUATE_TEMPLATE)


@app.post("/transcribe")
async def transcribe_upload(
    audio: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    decoder: str = Form(DEFAULT_DECODER),
    model: str = Form(DEFAULT_ENGINE),
) -> dict:
    require_engine(model, decoder, language)
    text = transcribe_bytes(await audio.read(), audio.filename or "audio.wav", language, decoder, model)
    return {"text": text, "language": language, "decoder": decoder if model == "indic" else "", "model": model}


@app.post("/evaluate")
async def evaluate_upload(
    audio: UploadFile = File(...),
    reference: str = Form(""),
    language: str = Form(DEFAULT_LANGUAGE),
    decoder: str = Form(DEFAULT_DECODER),
    model: str = Form(DEFAULT_ENGINE),
) -> dict:
    require_engine(model, decoder, language)
    hypothesis = transcribe_bytes(await audio.read(), audio.filename or "audio.wav", language, decoder, model)
    return {
        "filename": audio.filename or "audio.wav",
        "language": language,
        "model": model,
        "decoder": decoder if model == "indic" else "",
        "hypothesis": hypothesis,
        "reference": reference.strip(),
        "character_error_rate": error_rate(reference, hypothesis, by_word=False),
        "word_error_rate": error_rate(reference, hypothesis, by_word=True),
        "diff": diff_spans(reference, hypothesis),
    }
