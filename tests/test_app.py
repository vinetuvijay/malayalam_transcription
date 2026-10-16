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


def test_health_skips_model_load(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "loading"
    assert body["model"] == main.MODEL_ID
    assert body["google"] == "missing"


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
    assert response.json() == {"text": "നമസ്കാരം", "language": "ml", "decoder": "ctc", "model": "indic"}


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


def test_diff_marks_only_the_changed_characters():
    diff = main.diff_spans("മറ്റ്", "മറ്റു")
    assert diff["reference"] == [
        {"kind": "equal", "text": "മറ്റ"},
        {"kind": "change", "text": "്"},
    ]
    assert diff["hypothesis"] == [
        {"kind": "equal", "text": "മറ്റ"},
        {"kind": "change", "text": "ു"},
    ]


def test_diff_is_empty_without_a_reference():
    assert main.diff_spans("  ", "ഹലോ") is None


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
    assert body["diff"]["reference"][-1]["kind"] == "equal"
    assert any(span["kind"] == "change" for span in body["diff"]["hypothesis"])


def test_index_offers_google(client):
    response = client.get("/")
    assert "Google" in response.text
    assert "IndicConformer" in response.text


def test_google_transcribe_skips_the_local_model(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main, "export_16k_wav", lambda src: src)
    monkeypatch.setattr(main, "transcribe_gemini", lambda path, language: "ഹലോ")
    main.state["model"] = None
    main.state["loading"] = True

    response = client.post(
        "/transcribe",
        files={"audio": ("clip.wav", b"RIFFxxxx", "audio/wav")},
        data={"model": "google", "language": "ml"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "ഹലോ", "language": "ml", "decoder": "", "model": "google"}


def test_google_without_a_key_explains_how_to_set_it(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post(
        "/transcribe",
        files={"audio": ("clip.wav", b"RIFFxxxx", "audio/wav")},
        data={"model": "google", "language": "ml"},
    )
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_transcribe_rejects_an_unknown_model(client):
    response = client.post(
        "/transcribe",
        files={"audio": ("clip.wav", b"RIFFxxxx", "audio/wav")},
        data={"model": "whisper", "language": "ml"},
    )
    assert response.status_code == 400


def test_gemini_transcript_joins_text_parts():
    payload = {"candidates": [{"content": {"parts": [{"text": "ഹലോ"}, {"text": "നമസ്കാരം"}]}}]}
    assert main.gemini_transcript(payload) == "ഹലോ നമസ്കാരം"


def test_gemini_rate_limit_mentions_the_free_tier():
    error = main.gemini_failure(429, b'{"error":{"message":"quota"}}')
    assert error.status_code == 429
    assert "free limit" in error.detail


def test_transcribe_gemini_asks_for_malayalam(tmp_path, monkeypatch):
    wav_path = tmp_path / "clip.wav"
    with main.wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)

    calls = []

    class _Response:
        def __init__(self, payload, headers=None):
            self.payload = json_payload(payload)
            self.headers = {key.lower(): value for key, value in (headers or {}).items()}

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def json_payload(payload):
        import json

        return json.dumps(payload).encode()

    def fake_urlopen(request, timeout):
        calls.append(request)
        if request.method == "DELETE":
            return _Response({})
        if request.full_url.endswith("/upload/v1beta/files"):
            return _Response({}, {"x-goog-upload-url": "https://example.test/upload/abc"})
        if request.full_url == "https://example.test/upload/abc":
            return _Response(
                {
                    "file": {
                        "name": "files/abc",
                        "uri": "https://example.test/files/abc",
                        "mimeType": "audio/wav",
                        "state": "ACTIVE",
                    }
                }
            )
        return _Response({"candidates": [{"content": {"parts": [{"text": "ഹലോ"}]}}]})

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main.urllib.request, "urlopen", fake_urlopen)

    assert main.transcribe_gemini(wav_path, "ml") == "ഹലോ"
    generate = next(request for request in calls if "generateContent" in request.full_url)
    import json

    body = json.loads(generate.data.decode())
    assert "gemini-3.5-transcribe" in generate.full_url
    assert body["generationConfig"]["audioTranscriptionConfig"]["languageCodes"] == ["ml-IN"]
    start = next(request for request in calls if request.full_url.endswith("/upload/v1beta/files"))
    assert start.get_header("X-goog-upload-command") == "start"
    assert "test-key" not in generate.full_url
    assert generate.get_header("X-goog-api-key") == "test-key"
    assert any(request.method == "DELETE" for request in calls)


def test_transcribe_gemini_rejects_audio_longer_than_an_hour(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "wav_seconds", lambda path: 3601)
    with pytest.raises(main.HTTPException) as caught:
        main.transcribe_gemini(tmp_path / "long.wav", "ml")
    assert caught.value.status_code == 400


def test_evaluate_can_use_google(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main, "export_16k_wav", lambda src: src)
    monkeypatch.setattr(main, "transcribe_gemini", lambda path, language: "ഹലോ")
    main.state["model"] = None
    main.state["loading"] = True

    response = client.post(
        "/evaluate",
        files={"audio": ("call.wav", b"RIFFxxxx", "audio/wav")},
        data={"model": "google", "language": "ml", "reference": "ഹലോ"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "google"
    assert body["hypothesis"] == "ഹലോ"
    assert body["character_error_rate"] == 0


def test_gated_download_error_mentions_the_token():
    message = main.friendly_load_error(RuntimeError("401 Unauthorized"))
    assert "HF_TOKEN" in message
