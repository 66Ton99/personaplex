"""Pre-tokenize stereo Moshi training data for the native MLX trainer."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np
import rustymimi
import sentencepiece
import sphn


SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
FRAME_SAMPLES = int(SAMPLE_RATE / FRAME_RATE)
# Larger encode_step calls reduce Python overhead while remaining bit-identical
# to the 80 ms streaming runtime input path.
MIMI_STREAM_BLOCK_FRAMES = 125
TEXT_PADDING_ID = 3
END_OF_TEXT_PADDING_ID = 0
MAIN_SPEAKER = "SPEAKER_MAIN"


def _audio_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return manifest.parent / path


def _resolve_tokenizer(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    candidates = [
        Path("model-mlx/tokenizer_spm_32k_3.model"),
        Path("model/tokenizer_spm_32k_3.model"),
    ]
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    candidates.extend(
        hf_home.glob(
            "hub/models--nvidia--personaplex-7b-v1/snapshots/*/tokenizer_spm_32k_3.model"
        )
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Pass --tokenizer tokenizer_spm_32k_3.model")


def _build_text_stream(
    alignments: list[list[object]],
    tokenizer: sentencepiece.SentencePieceProcessor,
    frame_count: int,
) -> np.ndarray:
    main = sorted(
        (item for item in alignments if len(item) == 3 and item[2] == MAIN_SPEAKER),
        key=lambda item: float(item[1][0]),  # type: ignore[index]
    )
    tokenized = [
        (tokenizer.encode(str(item[0]).strip()), tuple(map(float, item[1])))
        for item in main
    ]
    stream = np.full(frame_count, TEXT_PADDING_ID, dtype=np.int32)
    queue: deque[int] = deque()
    alignment_index = 0
    last_word_end = -1

    for frame in range(frame_count):
        while (
            alignment_index < len(tokenized)
            and tokenized[alignment_index][1][0] * FRAME_RATE < frame + 1
        ):
            tokens, timestamp = tokenized[alignment_index]
            # Common Voice word boundaries can be denser than one token per
            # 80 ms frame. Queue them instead of dropping the later word.
            queue.extend(tokens)
            last_word_end = int(timestamp[1] * FRAME_RATE)
            alignment_index += 1

        if queue:
            if frame > 0 and stream[frame - 1] == TEXT_PADDING_ID:
                stream[frame - 1] = END_OF_TEXT_PADDING_ID
            stream[frame] = queue.popleft()
        elif frame <= last_word_end:
            stream[frame] = TEXT_PADDING_ID

    if queue:
        raise ValueError(f"sample ends with {len(queue)} pending text tokens")
    return stream


def _load_manifest(path: Path, limit: int | None) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _prepare_session(
    index: int,
    row: dict[str, object],
    manifest: Path,
    output_dir: Path,
    split: str,
    mimi_path: Path,
    tokenizer_path: Path,
    stream_block_frames: int,
) -> dict[str, object]:
    audio_path = _audio_path(manifest, str(row["path"]))
    audio, sample_rate = sphn.read(str(audio_path))
    if sample_rate != SAMPLE_RATE:
        audio = sphn.resample(
            audio,
            src_sample_rate=sample_rate,
            dst_sample_rate=SAMPLE_RATE,
        )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"{audio_path}: expected stereo audio, got {audio.shape}")

    code_chunks = []
    audio_tokenizer = rustymimi.Tokenizer(str(mimi_path), num_codebooks=8)
    usable_samples = audio.shape[-1] // FRAME_SAMPLES * FRAME_SAMPLES
    block_samples = stream_block_frames * FRAME_SAMPLES
    for start in range(0, usable_samples, block_samples):
        chunk = np.ascontiguousarray(
            audio[:, None, start : min(usable_samples, start + block_samples)]
        )
        code_chunks.append(
            np.asarray(audio_tokenizer.encode_step(chunk), dtype=np.int32)
        )
    audio_codes = np.concatenate(code_chunks, axis=-1)
    expected_frames = usable_samples // FRAME_SAMPLES
    audio_codes = audio_codes[..., :expected_frames]
    if audio_codes.shape[:2] != (2, 8):
        raise ValueError(f"{audio_path}: unexpected Mimi codes {audio_codes.shape}")
    frame_count = audio_codes.shape[-1]

    metadata = json.loads(audio_path.with_suffix(".json").read_text(encoding="utf-8"))
    text_tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    text_codes = _build_text_stream(metadata["alignments"], text_tokenizer, frame_count)
    codes = np.concatenate(
        [text_codes[None], audio_codes.reshape(16, frame_count)],
        axis=0,
    ).astype(np.int32, copy=False)

    output_path = output_dir / f"{split}-{index:06d}.npz"
    np.savez_compressed(output_path, codes=codes)
    return {
        "path": str(output_path),
        "frames": frame_count,
        "source": str(audio_path.resolve()),
    }


def prepare(args: argparse.Namespace) -> None:
    rows = _load_manifest(args.manifest, args.limit)
    if not rows:
        raise SystemExit(f"No samples found in {args.manifest}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = output_dir / f"{args.split}.jsonl"
    output_rows = []
    completed_sources: set[str] = set()
    if args.resume and output_manifest.exists():
        output_rows = _load_manifest(output_manifest, None)
        completed_sources = {str(Path(str(row["source"])).resolve()) for row in output_rows}
    elif output_manifest.exists():
        output_manifest.unlink()
    rows = [
        row
        for row in rows
        if str(_audio_path(args.manifest, str(row["path"])).resolve()) not in completed_sources
    ]
    existing_indices = [int(Path(str(row["path"])).stem.rsplit("-", 1)[-1]) for row in output_rows]
    start_index = max(existing_indices, default=-1) + 1
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _prepare_session,
                start_index + offset,
                row,
                args.manifest,
                output_dir,
                args.split,
                args.mimi,
                args.tokenizer,
                args.stream_block_frames,
            ): offset
            for offset, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(futures):
            output_row = future.result()
            output_rows.append(output_row)
            with output_manifest.open("a", encoding="utf-8") as handle:
                json.dump(output_row, handle)
                handle.write("\n")
            completed += 1
            print(
                f"[{completed}/{len(rows)}] {Path(str(output_row['path'])).name}: "
                f"{output_row['frames']} frames"
            )

    print(f"Wrote {len(output_rows)} tokenized sessions to {output_manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/uk_UA/mlx-data"))
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--mimi", type=Path, default=Path("model-mlx/tokenizer.safetensors"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--stream-block-frames",
        type=int,
        default=MIMI_STREAM_BLOCK_FRAMES,
        help="Frames per stateful Mimi encode_step call (12.5 frames/second).",
    )
    args = parser.parse_args()
    args.tokenizer = _resolve_tokenizer(args.tokenizer)
    for path in (args.manifest, args.mimi, args.tokenizer):
        if not path.exists():
            parser.error(f"missing file: {path}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1 or args.stream_block_frames < 1:
        parser.error("--workers and --stream-block-frames must be positive")
    return args


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
