# Ukrainian Full-Duplex Adaptation

This workflow keeps PersonaPlex/Moshi as the full-duplex runtime model.
Common Voice provides real Ukrainian input speech, local `gpt-oss-120b`
provides constrained paraphrases, and Silero `v4_ua`/`mykyta` renders the
assistant channel. The text-only model and TTS are data-preparation tools only;
neither replaces PersonaPlex at runtime.

## Data layout

Download the official Ukrainian Common Voice 26.0 release from [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmqinqaqp00wunq07w9oyei38).
The repository includes a resumable downloader. Create an MDC API key, then run:

```bash
.venv-nix/bin/pip install datacollective
export MDC_API_KEY='...'
.venv-nix/bin/python scripts/download_common_voice_uk.py
```

Extract the resulting archive. The Ukrainian directory must contain `train.tsv`,
`dev.tsv`/`test.tsv`, and `clips/`. Common Voice is CC0, but its terms prohibit
speaker identification and re-hosting the corpus.

Each generated training WAV follows Moshi's channel convention:

- channel 0: Ukrainian assistant response;
- channel 1: original Ukrainian Common Voice clip from the user;
- adjacent JSON: timestamped assistant text labelled `SPEAKER_MAIN`.

The builder groups exchanges into 80-second sessions to match the training
profile. It also injects controlled overlap and user interruptions, so language
adaptation does not collapse into half-duplex turns.

Common Voice is read speech rather than conversation. Use it for language and
acoustic adaptation, then add real consented two-channel Ukrainian conversations
to recover natural interruptions, backchannels, and emotion.

## Build a bootstrap dataset

Load `openai/gpt-oss-120b` in LM Studio and start its local server on port 1234.
The builder downloads Silero through `torch.hub` once and then uses the cached
Ukrainian `mykyta` voice. Use `--workers 4` on a 12-performance-core Apple
Silicon machine; parallel and sequential synthesis are numerically identical.

Generate separate train and evaluation manifests from non-overlapping Common
Voice TSV splits. Do not use `validated.tsv` for training together with
`test.tsv`, because the validated set can contain test clips:

```bash
.venv-nix/bin/python scripts/build_common_voice_uk_dataset.py \
  training/uk_UA/raw/extracted/cv-corpus-26.0-2026-06-12/uk \
  --tsv train.tsv \
  --output-dir training/uk_UA/data-v3 \
  --split train \
  --limit 28387 \
  --seed 2026 \
  --resume \
  --tts-backend silero \
  --silero-speaker mykyta \
  --echo-probability 0.85 \
  --lm-studio-url http://127.0.0.1:1234/v1 \
  --model openai/gpt-oss-120b \
  --overlap-probability 0.15 \
  --interruption-probability 0.15 \
  --workers 4

.venv-nix/bin/python scripts/build_common_voice_uk_dataset.py \
  training/uk_UA/raw/extracted/cv-corpus-26.0-2026-06-12/uk \
  --tsv dev.tsv \
  --output-dir training/uk_UA/data-v3-eval \
  --split eval \
  --limit 1000 \
  --seed 2027 \
  --tts-backend silero \
  --silero-speaker mykyta \
  --echo-probability 1 \
  --workers 4

.venv-nix/bin/python scripts/build_common_voice_uk_dataset.py \
  training/uk_UA/raw/extracted/cv-corpus-26.0-2026-06-12/uk \
  --tsv dev.tsv \
  --output-dir training/uk_UA/data-v3-eval \
  --split eval \
  --limit 1300 \
  --seed 2027 \
  --resume \
  --tts-backend silero \
  --silero-speaker mykyta \
  --echo-probability 0 \
  --model openai/gpt-oss-120b \
  --workers 4
```

The response cache and session manifest are append-only, so interrupted
generation can be resumed. Exact transcript-preserving responses dominate the
corpus because they provide the strongest supervision for Ukrainian acoustic
understanding; constrained natural paraphrases retain conversational behavior.
The builder pads TTS when Ukrainian byte-fallback tokens would exceed Moshi's
text frame rate.

## Validate

```bash
.venv-nix/bin/python scripts/validate_moshi_dataset.py \
  training/uk_UA/data-v3/train.jsonl
```

The validator reports `dropped`, the proportion of text tokens overwritten by
the stock 12.5 Hz interleaver. The MLX pre-tokenizer uses a FIFO queue instead,
so dense Ukrainian byte-fallback tokens are preserved rather than discarded.
Use the validator to detect malformed timings and channel layouts; verify the
pre-tokenized manifest separately before a long run.

## Train on macOS with MLX

The native MLX trainer keeps the base checkpoint frozen and trains LoRA adapters
on all 32 Temporal Transformer layers, the text input/output paths, and
DepFormer boundaries. It pre-tokenizes Mimi audio once, uses causal chunks with
gradient checkpointing, and evaluates every saved checkpoint on a fixed held-out
sample. Unload `gpt-oss-120b` from LM Studio and stop any old PersonaPlex server
after dataset generation so their unified-memory allocations do not compete
with training.

Verify Metal training and merged-checkpoint reload first:

```bash
.venv-nix/bin/python -m moshi.train_mlx --self-test
```

Convert the validated WAV sessions to compact 17-stream token files. The
pre-tokenizer is resumable and automatically uses up to eight performance cores
through the v3 training script:

```bash
.venv-nix/bin/python -m moshi.prepare_mlx_data \
  training/uk_UA/data-v3/train.jsonl \
  --output-dir training/uk_UA/mlx-data-v3 \
  --split train \
  --resume \
  --workers 8

.venv-nix/bin/python scripts/validate_mlx_tokens.py \
  training/uk_UA/mlx-data-v3/train.jsonl \
  --samples 5
```

The second command requires exact agreement with frame-by-frame RustyMimi
streaming. Offline chunk encoding with tokenizer resets is not equivalent to
the full-duplex runtime and must not be used for training.

The complete profile for the local M4 Max with 128 GB unified memory is:

```bash
scripts/train_ukrainian_v3_macos.sh
```

It trains rank-32 adapters with alpha 64, input-embedding adaptation, gradient
accumulation 4, a `3e-6` learning rate, 6,000 steps, evaluation every 250 steps,
and a merged MLX export. Gradient accumulation raises effective batch size
without retaining four activation graphs.

Resume both adapter weights and AdamW moments by setting the final target step:

```bash
.venv-nix/bin/python -m moshi.train_mlx \
  --data training/uk_UA/mlx-data-v3/train.jsonl \
  --eval-data training/uk_UA/mlx-data-v3-eval/eval.jsonl \
  --eval-batches 16 \
  --steps 8000 \
  --sequence-frames 128 \
  --grad-accum 4 \
  --rank 32 \
  --alpha 64 \
  --train-layers 32 \
  --adapt-input-embedding \
  --learning-rate 3e-6 \
  --save-every 250 \
  --memory-limit-gb 96 \
  --cache-limit-gb 4 \
  --output-dir training/uk_UA/mlx-runs/uk-v3 \
  --export-merged \
  --resume training/uk_UA/mlx-runs/uk-v3/adapter-006000.safetensors
```

Keep the same rank, alpha, layer count, and embedding flags when resuming so the
optimizer topology remains compatible. Increase the final target step rather
than restarting the step counter.

If an earlier adapter has the best held-out loss, merge it without adding more
training steps:

```bash
.venv-nix/bin/python -m moshi.train_mlx \
  --export-only \
  --resume training/uk_UA/mlx-runs/uk-v3/adapter-005250.safetensors \
  --rank 32 \
  --alpha 64 \
  --train-layers 32 \
  --adapt-input-embedding \
  --output-dir training/uk_UA/mlx-runs/uk-v3-selected
```

## Evaluate the merged MLX checkpoint

```bash
.venv-nix/bin/python scripts/evaluate_ukrainian_mlx.py \
  --weights training/uk_UA/mlx-runs/uk-v3/model.moshi_mlx.safetensors \
  --manifest training/uk_UA/data-v3-eval/eval.jsonl \
  --output-dir training/uk_UA/eval-runs/uk-v3 \
  --limit 20 \
  --enforce
```

The evaluator writes per-case Ukrainian text and WAV files plus `summary.json`
with Ukrainian-letter ratio, held-out content-stem recall, real-time factor,
response latency relative to the end of user speech, audio peak/RMS, clipping,
and an enforced pass/fail decision. Do not promote a checkpoint based only on
training loss.

Run an independent Whisper round-trip gate to verify that the generated audio
is intelligible and matches the text PersonaPlex intended to speak:

```bash
.venv-nix/bin/pip install -e 'moshi[asr-eval]'
.venv-nix/bin/python scripts/score_ukrainian_audio_intelligibility.py \
  training/uk_UA/eval-runs/uk-v3/results.jsonl \
  --model openai/whisper-small \
  --enforce
```

`--export-merged` writes `model.moshi_mlx.safetensors` and `config.json`. Run the
accepted checkpoint with the existing Mimi/tokenizer assets:

```bash
SSL_DIR=$(mktemp -d); python -m moshi.server \
  --ssl "$SSL_DIR" \
  --device mps \
  --mlx-model-dir model-mlx \
  --moshi-weight training/uk_UA/mlx-runs/uk-v3/model.moshi_mlx.safetensors \
  --mlx-default-voice off
```

## Fine-tune on CUDA

Use the official [Moshi fine-tuning trainer](https://github.com/kyutai-labs/moshi-finetune);
this repository only contains inference code:

```bash
git clone https://github.com/kyutai-labs/moshi-finetune.git ../moshi-finetune
uv run --project ../moshi-finetune \
  torchrun --nproc-per-node 1 -m train \
  training/uk_UA/personaplex_uk.yaml
```

The supplied CUDA profile uses LoRA plus embedding fine-tuning and writes merged
checkpoints. The official reference reports roughly
40 GB peak memory for a 7B LoRA run with larger batches, so adjust
`duration_sec`, `batch_size`, and `num_microbatches` for the available GPU.

## Run the checkpoint

Copy or point `--moshi-weight` at the merged `consolidated.safetensors` file and
keep the original Mimi and SentencePiece files:

```bash
python -m moshi.server \
  --device cuda \
  --moshi-weight /path/to/consolidated.safetensors \
  --mimi-weight model/tokenizer.safetensors \
  --tokenizer /path/to/tokenizer_spm_32k_3.model \
  --voice-prompt-dir model/voices
```

For Apple Silicon inference, convert the merged checkpoint with
`python -m moshi.convert_mlx` before starting the server with `--device mps`.
