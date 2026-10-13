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


def test_gated_download_error_mentions_the_token():
    message = main.friendly_load_error(RuntimeError("401 Unauthorized"))
    assert "HF_TOKEN" in message
