import os

os.environ["SKIP_MODEL_LOAD"] = "1"

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture
def client():
    main.state["model"] = None
    main.state["error"] = None
    main.state["loading"] = True
    with TestClient(main.app) as test_client:
        yield test_client


def test_index_has_upload_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Malayalam transcription" in response.text
    assert 'id="form"' in response.text
    assert "Start recording" in response.text


def test_health_skips_model_load(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "loading"
    assert response.json()["model"] == main.MODEL_ID


def test_transcribe_requires_a_loaded_model(client):
    main.state["model"] = None
    main.state["loading"] = True
    main.state["error"] = None
    response = client.post(
        "/transcribe",
        files={"audio": ("clip.wav", b"not-empty", "audio/wav")},
        data={"decoder": "rnnt", "language": "ml"},
    )
    assert response.status_code == 503


def test_transcribe_returns_text(client, monkeypatch):
    main.state["model"] = object()
    main.state["loading"] = False
    main.state["error"] = None
    monkeypatch.setattr(main, "to_16k_mono", lambda src: object())
    monkeypatch.setattr(main, "transcribe_wav", lambda wav, language, decoder: "നമസ്കാരം")

    response = client.post(
        "/transcribe",
        files={"audio": ("clip.wav", b"RIFFxxxx", "audio/wav")},
        data={"decoder": "ctc", "language": "ml"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "നമസ്കാരം", "language": "ml", "decoder": "ctc"}


def test_transcribe_rejects_an_unknown_decoder(client):
    main.state["model"] = object()
    main.state["loading"] = False
    main.state["error"] = None
    response = client.post(
        "/transcribe",
        files={"audio": ("clip.wav", b"RIFFxxxx", "audio/wav")},
        data={"decoder": "beam", "language": "ml"},
    )
    assert response.status_code == 400


def test_upload_limit_is_100_mb():
    assert main.MAX_BYTES == 100 * 1024 * 1024


def test_chunk_ranges_split_a_long_recording():
    ranges = main.chunk_ranges(16000 * 150)
    assert ranges == [(0, 16000 * 60), (16000 * 60, 16000 * 120), (16000 * 120, 16000 * 150)]


def test_chunk_ranges_keep_a_short_recording_whole():
    assert main.chunk_ranges(16000 * 20) == [(0, 16000 * 20)]


def test_chunk_ranges_drop_a_tiny_tail():
    ranges = main.chunk_ranges(16000 * 60 + 1000)
    assert ranges == [(0, 16000 * 60)]


class _FakeWav:
    def __init__(self, total):
        self.shape = (1, total)

    def __getitem__(self, item):
        span = item[1]
        return _FakeWav(span.stop - span.start)


def test_transcribe_wav_joins_chunk_text(monkeypatch):
    pytest.importorskip("torch")
    seen = []

    def fake_model(wav, language, decoder):
        seen.append((wav.shape[-1], language, decoder))
        return f"ഭാഗം{len(seen)}"

    main.state["model"] = fake_model
    text = main.transcribe_wav(_FakeWav(16000 * 90), "ml", "rnnt")
    assert seen == [(16000 * 60, "ml", "rnnt"), (16000 * 30, "ml", "rnnt")]
    assert text == "ഭാഗം1 ഭാഗം2"


def test_character_error_rate_is_zero_for_a_match():
    assert main.error_rate("നമസ്കാരം", "നമസ്കാരം", by_word=False) == 0


def test_character_error_rate_counts_a_substitution():
    assert main.error_rate("ഹലോ", "ഹലാ", by_word=False) == 0.3333


def test_error_rate_without_a_reference_is_empty():
    assert main.error_rate("   ", "ഹലോ", by_word=False) is None
    assert main.error_rate("", "ഹലോ", by_word=True) is None


def test_evaluate_page_shows_the_runner(client):
    response = client.get("/evaluate")
    assert response.status_code == 200
    assert "Run evaluation" in response.text
    assert "Expected Malayalam" in response.text


def test_evaluate_returns_malayalam_and_the_error_rate(client, monkeypatch):
    main.state["model"] = object()
    main.state["loading"] = False
    main.state["error"] = None
    monkeypatch.setattr(main, "to_16k_mono", lambda src: object())
    monkeypatch.setattr(main, "transcribe_wav", lambda wav, language, decoder: "ഹലോ ഹലോ")

    response = client.post(
        "/evaluate",
        files={"audio": ("call.wav", b"RIFFxxxx", "audio/wav")},
        data={"decoder": "rnnt", "language": "ml", "reference": "ഹലോ"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "call.wav"
    assert body["hypothesis"] == "ഹലോ ഹലോ"
    assert body["reference"] == "ഹലോ"
    assert body["character_error_rate"] == 1.3333
    assert body["word_error_rate"] == 1.0


def test_gated_download_error_mentions_the_token():
    message = main.friendly_load_error(RuntimeError("401 Unauthorized"))
    assert "HF_TOKEN" in message
