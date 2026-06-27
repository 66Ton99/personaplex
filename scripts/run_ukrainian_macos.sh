#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv-nix/bin/python"
MODEL_WEIGHT="$ROOT_DIR/training/uk_UA/mlx-runs/uk-clean/model.moshi_mlx.safetensors"
PORT="${PORT:-8998}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_WEIGHT" ]]; then
  echo "Missing Ukrainian model: $MODEL_WEIGHT" >&2
  exit 1
fi

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/moshi${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m moshi.server \
  --device mps \
  --mlx-model-dir "$ROOT_DIR/model-mlx" \
  --moshi-weight "$MODEL_WEIGHT" \
  --mlx-default-voice off \
  --host 127.0.0.1 \
  --port "$PORT"
