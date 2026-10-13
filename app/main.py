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
TEMPLATE = Path(__file__).parent / "templates" / "index.html"
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
            timeout=120,
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


def transcribe_wav(wav, language: str, decoder: str) -> str:
    import torch

    model = state["model"]
    with infer_lock:
        with torch.inference_mode():
            return as_text(model(wav, language, decoder))


@app.get("/health")
def health() -> dict:
    if state["error"]:
        return {"status": "error", "detail": state["error"], "model": MODEL_ID}
    if state["loading"] or state["model"] is None:
        return {"status": "loading", "model": MODEL_ID}
    return {"status": "ready", "model": MODEL_ID, "language": DEFAULT_LANGUAGE}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(TEMPLATE)


@app.post("/transcribe")
async def transcribe_upload(
    audio: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    decoder: str = Form(DEFAULT_DECODER),
) -> dict:
    if decoder not in {"ctc", "rnnt"}:
        raise HTTPException(status_code=400, detail="Decoder must be ctc or rnnt.")
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language id.")
    if state["error"]:
        raise HTTPException(status_code=503, detail=state["error"])
    if state["loading"] or state["model"] is None:
        raise HTTPException(status_code=503, detail="The model is still downloading.")

    payload = await audio.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The audio file is empty.")
    if len(payload) > MAX_BYTES:
        limit_mb = MAX_BYTES // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"The file is larger than {limit_mb} MB.")

    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".audio"

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{suffix}"
        src.write_bytes(payload)
        wav = to_16k_mono(src)
        text = transcribe_wav(wav, language, decoder)

    return {"text": text, "language": language, "decoder": decoder}
