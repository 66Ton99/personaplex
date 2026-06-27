#!/usr/bin/env python3
"""Measure generated Ukrainian speech intelligibility with local Whisper ASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import time

import numpy as np
import sphn


SAMPLE_RATE = 16_000
WORD_RE = re.compile(r"[0-9A-Za-zА-ЩЬЮЯЄІЇҐа-щьюяєіїґ]+")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("'", ""))


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = _words(reference)
    hypothesis_words = _words(hypothesis)
    return _edit_distance(reference_words, hypothesis_words) / max(1, len(reference_words))


def _load_results(path: Path, limit: int | None) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("output_wav") and row.get("response_text"):
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def score(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    rows = _load_results(args.results, args.limit)
    if not rows:
        raise SystemExit(f"No generated WAV/text pairs found in {args.results}")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Loading {args.model} on {device}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ukrainian",
        task="transcribe",
    )

    output_path = args.output or args.results.with_name("asr-results.jsonl")
    scored: list[dict[str, object]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            audio, sample_rate = sphn.read(str(row["output_wav"]))
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=0)
            if sample_rate != SAMPLE_RATE:
                audio = np.asarray(
                    sphn.resample(
                        audio[None],
                        src_sample_rate=sample_rate,
                        dst_sample_rate=SAMPLE_RATE,
                    )
                ).reshape(-1)
            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            features = inputs.input_features.to(device=device, dtype=dtype)
            attention_mask = inputs.attention_mask.to(device=device)
            started = time.perf_counter()
            with torch.inference_mode():
                token_ids = model.generate(
                    features,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                )
            hypothesis = processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
            reference = str(row["response_text"])
            wer = _word_error_rate(reference, hypothesis)
            result = {
                "output_wav": row["output_wav"],
                "reference_text": reference,
                "asr_text": hypothesis,
                "word_error_rate": round(wer, 4),
                "seconds": round(time.perf_counter() - started, 3),
            }
            scored.append(result)
            json.dump(result, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(rows)}] WER={wer:.3f} {hypothesis}",
                flush=True,
            )

    wers = [float(row["word_error_rate"]) for row in scored]
    mean_wer = float(np.mean(wers))
    failures = []
    if mean_wer > args.max_mean_wer:
        failures.append("mean Whisper word error rate exceeds threshold")
    summary = {
        "cases": len(scored),
        "model": args.model,
        "mean_word_error_rate": round(mean_wer, 4),
        "median_word_error_rate": round(statistics.median(wers), 4),
        "max_word_error_rate": round(max(wers), 4),
        "passed": not failures,
        "failures": failures,
    }
    summary_path = output_path.with_name("asr-summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.enforce and failures:
        raise SystemExit("Ukrainian audio intelligibility gate failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results.jsonl from evaluate_ukrainian_mlx.py")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-mean-wer", type=float, default=0.45)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if not args.results.exists():
        parser.error(f"missing results file: {args.results}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1 or args.max_mean_wer < 0:
        parser.error("token count must be positive and WER threshold non-negative")
    return args


if __name__ == "__main__":
    score(parse_args())
