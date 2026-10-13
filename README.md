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

Audio is converted to 16 kHz mono before inference. Uploads are limited to 100 MB. Recordings longer than one minute are split into 60-second pieces and the text is joined. RNNT is the default decoder; CTC is the faster option on CPU. Plan on several gigabytes of free RAM. A long recording can take several minutes to transcribe.

`GET /health` reports `loading`, `ready`, or `error` while the model download finishes.

## Evaluation

Open http://localhost:8000/evaluate. Add one or more clips and, when you have it, paste the Malayalam that should be written. Each result shows the transcription. A reference also produces character and word error rates.

Google FLEURS already pairs each Malayalam recording with its transcript. This saves the first 20 test clips, the normalized transcript, and the original punctuated sentence:

```bash
pip install datasets soundfile
python download_eval.py --limit 20
```

`data/fleurs-ml/0000.wav` is the audio. The same row in `data/fleurs-ml/references.tsv` is the Malayalam to paste into **Expected Malayalam**. FLEURS is published under CC BY 4.0.
