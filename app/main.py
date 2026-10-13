import os
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
DEFAULT_LANGUAGE = os.environ.get("LANGUAGE", "ml")
DEFAULT_DECODER = os.environ.get("DECODER", "rnnt")
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


def to_16k_mono(src: Path):
    import soundfile as sf
    import torch

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


def transcription_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) > 300:
        message = message[:300] + "…"
    return f"Transcription failed: {message}"


@app.get("/health")
def health() -> dict:
    if state["error"]:
        return {"status": "error", "detail": state["error"], "model": MODEL_ID}
    if state["loading"] or state["model"] is None:
        return {"status": "loading", "model": MODEL_ID}
    return {"status": "ready", "model": MODEL_ID, "language": DEFAULT_LANGUAGE}


def require_model(decoder: str, language: str) -> None:
    if decoder not in {"ctc", "rnnt"}:
        raise HTTPException(status_code=400, detail="Decoder must be ctc or rnnt.")
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language id.")
    if state["error"]:
        raise HTTPException(status_code=503, detail=state["error"])
    if state["loading"] or state["model"] is None:
        raise HTTPException(status_code=503, detail="The model is still downloading.")


def transcribe_bytes(payload: bytes, filename: str, language: str, decoder: str) -> str:
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
        wav = to_16k_mono(src)
        try:
            return transcribe_wav(wav, language, decoder)
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
) -> dict:
    require_model(decoder, language)
    text = transcribe_bytes(await audio.read(), audio.filename or "audio.wav", language, decoder)
    return {"text": text, "language": language, "decoder": decoder}


@app.post("/evaluate")
async def evaluate_upload(
    audio: UploadFile = File(...),
    reference: str = Form(""),
    language: str = Form(DEFAULT_LANGUAGE),
    decoder: str = Form(DEFAULT_DECODER),
) -> dict:
    require_model(decoder, language)
    hypothesis = transcribe_bytes(await audio.read(), audio.filename or "audio.wav", language, decoder)
    return {
        "filename": audio.filename or "audio.wav",
        "language": language,
        "decoder": decoder,
        "hypothesis": hypothesis,
        "reference": reference.strip(),
        "character_error_rate": error_rate(reference, hypothesis, by_word=False),
        "word_error_rate": error_rate(reference, hypothesis, by_word=True),
    }
