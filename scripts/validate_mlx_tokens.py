#!/usr/bin/env python3
"""Validate pre-tokenized PersonaPlex data against streaming RustyMimi."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rustymimi
import sphn


SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1_920
AUDIO_CODEBOOKS = 8
EXPECTED_STREAMS = 17


def _load_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sample_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def _stream_reference(audio: np.ndarray, mimi: Path, frames: int) -> np.ndarray:
    tokenizer = rustymimi.Tokenizer(str(mimi), num_codebooks=AUDIO_CODEBOOKS)
    chunks = []
    for frame in range(frames):
        start = frame * FRAME_SAMPLES
        chunk = np.ascontiguousarray(audio[:, None, start : start + FRAME_SAMPLES])
        if chunk.shape[-1] != FRAME_SAMPLES:
            break
        chunks.append(np.asarray(tokenizer.encode_step(chunk), dtype=np.int32))
    if not chunks:
        raise ValueError("audio is shorter than one Mimi frame")
    return np.concatenate(chunks, axis=-1)


def validate(args: argparse.Namespace) -> None:
    rows = _load_rows(args.manifest)
    if not rows:
        raise SystemExit(f"Empty token manifest: {args.manifest}")
    paths = [str(Path(str(row["path"])).resolve()) for row in rows]
    sources = [str(Path(str(row["source"])).resolve()) for row in rows]
    if len(paths) != len(set(paths)):
        raise SystemExit("Token manifest contains duplicate output paths")
    if len(sources) != len(set(sources)):
        raise SystemExit("Token manifest contains duplicate source sessions")

    indices = _sample_indices(len(rows), min(args.samples, len(rows)))
    for position, index in enumerate(indices, start=1):
        row = rows[index]
        token_path = Path(str(row["path"]))
        source_path = Path(str(row["source"]))
        with np.load(token_path) as archive:
            codes = np.asarray(archive["codes"], dtype=np.int32)
        if codes.ndim != 2 or codes.shape[0] != EXPECTED_STREAMS:
            raise SystemExit(f"{token_path}: expected 17 streams, got {codes.shape}")
        if codes.shape[1] != int(row["frames"]):
            raise SystemExit(
                f"{token_path}: manifest says {row['frames']} frames, got {codes.shape[1]}"
            )
        audio, sample_rate = sphn.read(str(source_path))
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != SAMPLE_RATE or audio.ndim != 2 or audio.shape[0] != 2:
            raise SystemExit(
                f"{source_path}: expected stereo {SAMPLE_RATE} Hz audio, got {audio.shape} at {sample_rate}"
            )
        frames = min(args.frames, codes.shape[1])
        reference = _stream_reference(audio, args.mimi, frames)
        actual = codes[1:].reshape(2, AUDIO_CODEBOOKS, -1)[..., : reference.shape[-1]]
        if not np.array_equal(reference, actual):
            agreement = float(np.mean(reference == actual))
            raise SystemExit(
                f"{token_path}: streaming Mimi mismatch, token agreement={agreement:.6f}"
            )
        print(f"[{position}/{len(indices)}] {token_path.name}: streaming match")

    print(
        f"validated rows={len(rows)} unique_sources={len(set(sources))} "
        f"streaming_samples={len(indices)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mimi", type=Path, default=Path("model-mlx/tokenizer.safetensors"))
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--frames", type=int, default=125)
    args = parser.parse_args()
    for path in (args.manifest, args.mimi):
        if not path.exists():
            parser.error(f"missing file: {path}")
    if args.samples < 1 or args.frames < 1:
        parser.error("--samples and --frames must be positive")
    return args


if __name__ == "__main__":
    validate(parse_args())
