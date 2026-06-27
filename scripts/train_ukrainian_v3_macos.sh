#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv-nix/bin/python"
TOKENIZER="${TOKENIZER:-$HOME/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/3343b641d663e4c851120b3575cbdfa4cc33e7fa/tokenizer_spm_32k_3.model}"
TRAIN_DATA="$ROOT_DIR/training/uk_UA/data-v3/train.jsonl"
EVAL_DATA="$ROOT_DIR/training/uk_UA/data-v3-eval/eval.jsonl"
TRAIN_TOKENS="$ROOT_DIR/training/uk_UA/mlx-data-v3"
EVAL_TOKENS="$ROOT_DIR/training/uk_UA/mlx-data-v3-eval"
OUTPUT_DIR="$ROOT_DIR/training/uk_UA/mlx-runs/uk-v3"

if [[ -z "${PERSONAPLEX_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -ims env PERSONAPLEX_CAFFEINATED=1 "$0" "$@"
fi

if [[ -z "${MIMI_WORKERS:-}" ]]; then
  PERF_CORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.ncpu)"
  if (( PERF_CORES > 8 )); then
    MIMI_WORKERS=8
  else
    MIMI_WORKERS="$PERF_CORES"
  fi
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/moshi${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" scripts/validate_moshi_dataset.py "$TRAIN_DATA" --tokenizer "$TOKENIZER"
"$PYTHON_BIN" scripts/validate_moshi_dataset.py "$EVAL_DATA" --tokenizer "$TOKENIZER"

"$PYTHON_BIN" -m moshi.prepare_mlx_data \
  "$TRAIN_DATA" \
  --output-dir "$TRAIN_TOKENS" \
  --split train \
  --resume \
  --workers "$MIMI_WORKERS"

"$PYTHON_BIN" -m moshi.prepare_mlx_data \
  "$EVAL_DATA" \
  --output-dir "$EVAL_TOKENS" \
  --split eval \
  --resume \
  --workers "$MIMI_WORKERS"

"$PYTHON_BIN" scripts/validate_mlx_tokens.py \
  "$TRAIN_TOKENS/train.jsonl" \
  --samples 5

"$PYTHON_BIN" scripts/validate_mlx_tokens.py \
  "$EVAL_TOKENS/eval.jsonl" \
  --samples 3

resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "$RESUME")
fi

exec "$PYTHON_BIN" -m moshi.train_mlx \
  --data "$TRAIN_TOKENS/train.jsonl" \
  --eval-data "$EVAL_TOKENS/eval.jsonl" \
  --eval-batches 16 \
  --steps "${STEPS:-6000}" \
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
  --output-dir "$OUTPUT_DIR" \
  --export-merged \
  "${resume_args[@]}"
