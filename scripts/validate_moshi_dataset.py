#!/usr/bin/env python3
"""Validate stereo Moshi data and detect text tokens lost by interleaving."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sentencepiece


FRAME_RATE = 12.5
CYRILLIC_RE = re.compile(r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]")


@dataclass
class Report:
    files: int = 0
    duration: float = 0.0
    words: int = 0
    tokens: int = 0
    byte_tokens: int = 0
    dropped_tokens: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resolve_audio_path(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path
    return manifest.parent / path


def _resolve_tokenizer(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    candidates = [Path("model/tokenizer_spm_32k_3.model")]
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


def _simulate_interleaver(
    alignments: list[list[Any]], tokenizer: Any, duration: float
) -> tuple[int, int, int, int]:
    main = sorted(
        (item for item in alignments if len(item) == 3 and item[2] == "SPEAKER_MAIN"),
        key=lambda item: item[1][0],
    )
    tokenized = [(tokenizer.encode(item[0].strip()), item[1]) for item in main]
    queue: deque[int] = deque()
    index = 0
    dropped = 0
    emitted = 0
    byte_tokens = 0
    total_tokens = 0
    for tokens, _ in tokenized:
        total_tokens += len(tokens)
        byte_tokens += sum(tokenizer.id_to_piece(token).startswith("<0x") for token in tokens)

    for frame in range(math.ceil(duration * FRAME_RATE)):
        while index < len(tokenized) and tokenized[index][1][0] * FRAME_RATE < frame + 1:
            if queue:
                dropped += len(queue)
            queue = deque(tokenized[index][0])
            index += 1
        if queue:
            queue.popleft()
            emitted += 1
    dropped += len(queue)
    return total_tokens, byte_tokens, dropped, emitted


def validate_dataset(manifest: Path, tokenizer: Any) -> Report:
    report = Report()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                audio_path = _resolve_audio_path(manifest, item["path"])
                declared_duration = float(item["duration"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                report.errors.append(f"{manifest}:{line_number}: invalid manifest row: {error}")
                continue

            if not audio_path.exists():
                report.errors.append(f"{manifest}:{line_number}: missing audio {audio_path}")
                continue
            try:
                with wave.open(str(audio_path), "rb") as audio:
                    channels = audio.getnchannels()
                    sample_rate = audio.getframerate()
                    sample_width = audio.getsampwidth()
                    duration = audio.getnframes() / sample_rate
            except (wave.Error, EOFError) as error:
                report.errors.append(f"{audio_path}: unreadable WAV: {error}")
                continue

            if channels != 2:
                report.errors.append(f"{audio_path}: expected 2 channels, found {channels}")
            if sample_rate != 24_000:
                report.errors.append(f"{audio_path}: expected 24000 Hz, found {sample_rate}")
            if sample_width != 2:
                report.warnings.append(f"{audio_path}: expected 16-bit PCM, found {sample_width * 8}-bit")
            if abs(duration - declared_duration) > 0.1:
                report.errors.append(
                    f"{audio_path}: manifest duration {declared_duration:.3f}s != WAV {duration:.3f}s"
                )

            metadata_path = audio_path.with_suffix(".json")
            if not metadata_path.exists():
                report.errors.append(f"{audio_path}: missing {metadata_path.name}")
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                alignments = metadata["alignments"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                report.errors.append(f"{metadata_path}: invalid alignments: {error}")
                continue

            previous_start = -1.0
            main_alignments = []
            valid_alignments = True
            for alignment_index, alignment in enumerate(alignments):
                try:
                    text, timestamp, speaker = alignment
                    start, end = map(float, timestamp)
                except (TypeError, ValueError) as error:
                    report.errors.append(
                        f"{metadata_path}: alignment {alignment_index} is invalid: {error}"
                    )
                    valid_alignments = False
                    continue
                if not text or not isinstance(speaker, str):
                    report.errors.append(
                        f"{metadata_path}: alignment {alignment_index} must contain text and a speaker label"
                    )
                    valid_alignments = False
                if speaker == "SPEAKER_MAIN":
                    main_alignments.append(alignment)
                if start < previous_start or start < 0 or end <= start or end > duration + 0.1:
                    report.errors.append(
                        f"{metadata_path}: invalid timestamp [{start}, {end}] at alignment {alignment_index}"
                    )
                    valid_alignments = False
                previous_start = start

            if not valid_alignments:
                continue
            if not main_alignments:
                report.errors.append(f"{metadata_path}: no SPEAKER_MAIN alignments")
                continue
            text = " ".join(item[0] for item in main_alignments)
            if not CYRILLIC_RE.search(text):
                report.warnings.append(f"{metadata_path}: no Ukrainian Cyrillic text found")
            tokens, byte_tokens, dropped, _ = _simulate_interleaver(
                alignments, tokenizer, duration
            )
            if dropped:
                report.warnings.append(
                    f"{metadata_path}: {dropped}/{tokens} text tokens are overwritten at 12.5 Hz"
                )
            report.files += 1
            report.duration += duration
            report.words += len(main_alignments)
            report.tokens += tokens
            report.byte_tokens += byte_tokens
            report.dropped_tokens += dropped
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--strict", action="store_true", help="Fail on warnings too")
    parser.add_argument(
        "--verbose-warnings",
        action="store_true",
        help="Print every warning instead of a short sample.",
    )
    args = parser.parse_args()

    tokenizer_path = _resolve_tokenizer(args.tokenizer)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    report = validate_dataset(args.manifest, tokenizer)
    for error in report.errors:
        print(f"ERROR: {error}")
    visible_warnings = report.warnings if args.verbose_warnings else report.warnings[:10]
    for warning in visible_warnings:
        print(f"WARNING: {warning}")
    if len(visible_warnings) < len(report.warnings):
        print(f"WARNING: {len(report.warnings) - len(visible_warnings)} more warnings suppressed")
    byte_ratio = report.byte_tokens / report.tokens if report.tokens else 0.0
    dropped_ratio = report.dropped_tokens / report.tokens if report.tokens else 0.0
    print(
        f"files={report.files} hours={report.duration / 3600:.2f} words={report.words} "
        f"tokens={report.tokens} byte_tokens={byte_ratio:.1%} dropped={dropped_ratio:.1%} "
        f"warnings={len(report.warnings)}"
    )
    if report.errors or (args.strict and report.warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
