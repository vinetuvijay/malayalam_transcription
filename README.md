# Malayalam transcription

Local speech-to-text for Malayalam. IndicConformer (`ai4bharat/indic-conformer-600m-multilingual`) runs on this machine. Google Gemini 3.5 Transcribe is the other choice and sends the recording to Google.

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

Audio is converted to 16 kHz mono before inference. Uploads are limited to 100 MB. With IndicConformer, recordings longer than one minute are split into 60-second pieces and the text is joined. RNNT is the default decoder; CTC is the faster option on CPU. Plan on several gigabytes of free RAM. A long recording can take several minutes to transcribe.

Choose **Google** on the page to use Gemini 3.5 Transcribe (`gemini-3.5-transcribe`) with Malayalam (`ml-IN`). Create a free key in [Google AI Studio](https://aistudio.google.com/apikey) and set `GEMINI_API_KEY` in `.env`. The free tier has a daily limit. On that tier Google may use the audio to improve its products. Google accepts up to one hour per clip. IndicConformer does not need to finish downloading before a Google transcription.

`GET /health` reports `loading`, `ready`, or `error` while the IndicConformer download finishes, and `google` is `ready` or `missing`.

## Evaluation

Open http://localhost:8000/evaluate. Add one or more clips and, when you have it, paste the Malayalam that should be written. Each result plays the clip and shows the transcription. Characters that differ from the reference are highlighted, and the card shows character and word error rates.

Google FLEURS already pairs each Malayalam recording with its transcript. This saves the first 20 test clips, the normalized transcript, and the original punctuated sentence:

```bash
pip install datasets soundfile
python download_eval.py --limit 20
```

`data/fleurs-ml/0000.wav` is the audio. The same row in `data/fleurs-ml/references.tsv` is the Malayalam to paste into **Expected Malayalam**. FLEURS is published under CC BY 4.0.
