# Malayalam transcription

Local speech-to-text for Malayalam using AI4Bharat IndicConformer (`ai4bharat/indic-conformer-600m-multilingual`). The app serves a browser page where you can record or upload audio, then returns the transcription from the RNNT or CTC decoder.

## Run

The model is gated on Hugging Face. Accept the conditions on the [model page](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual), create a read token, and put it in `.env`:

```bash
cp .env.example .env
```

```bash
docker compose up --build
```

Open http://localhost:8000. The first start downloads the model into a Docker volume named `hf-cache`, so later starts reuse it.

```bash
curl -F "audio=@sample.wav" -F "decoder=rnnt" -F "language=ml" http://localhost:8000/transcribe
```

Audio is converted to 16 kHz mono before inference. Uploads are limited to 100 MB. RNNT is the default decoder; CTC is the faster option on CPU. Plan on several gigabytes of free RAM. A long recording can take several minutes to transcribe.

`GET /health` reports `loading`, `ready`, or `error` while the model download finishes.
