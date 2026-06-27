#!/usr/bin/env python3
"""Run held-out Ukrainian audio through a local PersonaPlex MLX checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import sentencepiece
import sphn


SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1_920
TEXT_PADDING_IDS = (0, 3)
MLX_SILENCE_TOKENS = (948, 243, 1178, 546, 1736, 1030, 1978, 2008)
MLX_SINE_TOKENS = (430, 1268, 381, 1611, 1095, 1495, 56, 472)
DEFAULT_PROMPT = (
    "Ти українськомовний голосовий асистент. Завжди уважно слухай співрозмовника "
    "і відповідай лише природною українською мовою. Не переходь на англійську."
)
UKRAINIAN_LETTER_RE = re.compile(r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]")
CONTENT_WORD_RE = re.compile(r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]{4,}")
STOPWORDS = frozenset(
    {
        "але",
        "вона",
        "вони",
        "воно",
        "його",
        "йому",
        "коли",
        "мене",
        "може",
        "нього",
        "після",
        "перед",
        "сама",
        "себе",
        "також",
        "тому",
        "через",
        "щось",
        "який",
        "яка",
        "яке",
        "якщо",
    }
)


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


def _wrap_system_prompt(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def _ukrainian_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(UKRAINIAN_LETTER_RE.fullmatch(char)) for char in letters) / len(letters)


def _content_stems(text: str) -> set[str]:
    return {
        word[:5]
        for word in CONTENT_WORD_RE.findall(text.lower())
        if word not in STOPWORDS
    }


def _load_cases(manifest: Path, limit: int) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = json.loads(Path(row["path"]).with_suffix(".json").read_text(encoding="utf-8"))
            for exchange in metadata["source"]["exchanges"]:
                source = str(Path(exchange["clip_path"]).resolve())
                if source in seen:
                    continue
                seen.add(source)
                cases.append(
                    {
                        "source": source,
                        "expected_text": exchange["user_text"],
                    }
                )
                if len(cases) >= limit:
                    return cases
    return cases


def _read_input(
    path: Path, lead_seconds: float, tail_seconds: float
) -> tuple[np.ndarray, float]:
    audio, sample_rate = sphn.read(str(path))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if sample_rate != SAMPLE_RATE:
        audio = np.asarray(
            sphn.resample(audio[None], src_sample_rate=sample_rate, dst_sample_rate=SAMPLE_RATE)
        ).reshape(-1)
    speech_end_seconds = lead_seconds + audio.size / SAMPLE_RATE
    lead = round(lead_seconds * SAMPLE_RATE)
    tail = round(tail_seconds * SAMPLE_RATE)
    return np.pad(audio, (lead, tail)), speech_end_seconds


def _reset_model_caches(model: object) -> None:
    for cache in [*model.transformer_cache, *model.depformer_cache]:  # type: ignore[attr-defined]
        cache.reset()


def _run_case(
    model: object,
    lm_config: object,
    text_tokenizer: sentencepiece.SentencePieceProcessor,
    mimi_path: Path,
    prompt_tokens: list[int],
    case: dict[str, str],
    args: argparse.Namespace,
    index: int,
) -> dict[str, object]:
    import mlx.core as mx
    import rustymimi
    from moshi_mlx import models, utils

    mx.random.seed(args.seed + index)
    _reset_model_caches(model)
    input_audio, speech_end_seconds = _read_input(
        Path(case["source"]), args.lead_seconds, args.tail_seconds
    )
    steps = input_audio.size // FRAME_SAMPLES
    audio_tokenizer = rustymimi.Tokenizer(
        str(mimi_path),
        num_codebooks=max(lm_config.generated_codebooks, lm_config.other_codebooks),
    )
    condition = (
        model.condition_provider.condition_tensor("description", "very_good")
        if model.condition_provider is not None
        else None
    )
    prompt_steps = len(prompt_tokens) + 12
    gen = models.LmGen(
        model=model,
        max_steps=steps + prompt_steps + 20,
        text_sampler=utils.Sampler(top_k=args.text_top_k, temp=args.temperature),
        audio_sampler=utils.Sampler(top_k=args.audio_top_k, temp=args.temperature),
        check=False,
    )

    sine = mx.array([MLX_SINE_TOKENS], dtype=mx.int32)

    def force_prompt_step(token: int) -> None:
        gen.step(sine, ct=condition)
        gen.gen_sequence[:, 0, gen.step_idx - 1] = token
        for codebook, delay in enumerate(gen.audio_delays[: gen.main_codebooks]):
            target = gen.step_idx - 1 - delay
            if target >= 0:
                gen.gen_sequence[:, codebook + 1, target] = MLX_SILENCE_TOKENS[codebook]

    for _ in range(6):
        force_prompt_step(3)
    for token in prompt_tokens:
        force_prompt_step(token)
    for _ in range(6):
        force_prompt_step(3)

    emitted_tokens: list[int] = []
    decoded_text = ""
    generated_audio: list[np.ndarray] = []
    first_text_step: int | None = None
    started = time.perf_counter()
    for step in range(steps):
        chunk = input_audio[step * FRAME_SAMPLES : (step + 1) * FRAME_SAMPLES]
        input_tokens = audio_tokenizer.encode_step(chunk[None, None])
        input_tokens = mx.array(input_tokens).transpose(0, 2, 1)[:, :, : lm_config.other_codebooks]
        text_token = int(gen.step(input_tokens[0], ct=condition)[0].item())
        if text_token not in TEXT_PADDING_IDS:
            if first_text_step is None:
                first_text_step = step
            emitted_tokens.append(text_token)
            current = text_tokenizer.decode(emitted_tokens)
            if "\ufffd" not in current:
                decoded_text = current
        output_tokens = gen.last_audio_tokens()
        if output_tokens is not None:
            output_tokens = np.asarray(output_tokens[:, :, None], dtype=np.uint32)
            generated_audio.append(np.asarray(audio_tokenizer.decode_step(output_tokens))[0, 0])
    elapsed = time.perf_counter() - started

    output_audio = (
        np.concatenate(generated_audio).astype(np.float32, copy=False)
        if generated_audio
        else np.zeros(0, dtype=np.float32)
    )
    stem = f"case-{index:03d}"
    wav_path = args.output_dir / f"{stem}.wav"
    text_path = args.output_dir / f"{stem}.txt"
    if output_audio.size:
        rustymimi.write_wav(str(wav_path), output_audio, sample_rate=SAMPLE_RATE)
    text_path.write_text(decoded_text + "\n", encoding="utf-8")

    expected_stems = _content_stems(case["expected_text"])
    response_stems = _content_stems(decoded_text)
    overlap = len(expected_stems & response_stems) / max(1, len(expected_stems))
    peak = float(np.max(np.abs(output_audio))) if output_audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(output_audio)))) if output_audio.size else 0.0
    clipping = float(np.mean(np.abs(output_audio) >= 0.999)) if output_audio.size else 0.0
    first_text_seconds = (
        first_text_step * FRAME_SAMPLES / SAMPLE_RATE if first_text_step is not None else None
    )
    response_latency_seconds = (
        first_text_seconds - speech_end_seconds if first_text_seconds is not None else None
    )
    return {
        **case,
        "response_text": decoded_text,
        "ukrainian_ratio": round(_ukrainian_ratio(decoded_text), 4),
        "content_stem_recall": round(overlap, 4),
        "first_text_seconds": first_text_seconds,
        "input_speech_end_seconds": round(speech_end_seconds, 4),
        "response_latency_seconds": (
            round(response_latency_seconds, 4)
            if response_latency_seconds is not None
            else None
        ),
        "realtime_factor": round(elapsed / max(0.001, steps * FRAME_SAMPLES / SAMPLE_RATE), 4),
        "audio_peak": round(peak, 6),
        "audio_rms": round(rms, 6),
        "audio_clipping_ratio": round(clipping, 8),
        "output_wav": str(wav_path) if output_audio.size else None,
        "output_text": str(text_path),
    }


def evaluate(args: argparse.Namespace) -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from moshi_mlx import models

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = json.loads(args.config.read_text(encoding="utf-8"))
    lm_config = models.LmConfig.from_config_dict(config_dict)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if args.quantized is not None:
        nn.quantize(model, bits=args.quantized, group_size=32 if args.quantized == 4 else 64)
    print(f"Loading {args.weights}")
    model.load_weights(str(args.weights), strict=True)
    condition = (
        model.condition_provider.condition_tensor("description", "very_good")
        if model.condition_provider is not None
        else None
    )
    model.warmup(condition)

    text_tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    prompt_tokens = text_tokenizer.encode(_wrap_system_prompt(args.prompt))
    cases = _load_cases(args.manifest, args.limit)
    if not cases:
        raise SystemExit(f"No evaluation cases found in {args.manifest}")

    results_path = args.output_dir / "results.jsonl"
    results: list[dict[str, object]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases):
            result = _run_case(
                model,
                lm_config,
                text_tokenizer,
                args.mimi,
                prompt_tokens,
                case,
                args,
                index,
            )
            results.append(result)
            json.dump(result, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            print(
                f"[{index + 1}/{len(cases)}] uk={result['ukrainian_ratio']:.2f} "
                f"recall={result['content_stem_recall']:.2f} {result['response_text']}"
            )

    response_latencies = [
        float(item["response_latency_seconds"])
        for item in results
        if item["response_latency_seconds"] is not None
    ]
    mean_ukrainian_ratio = float(np.mean([item["ukrainian_ratio"] for item in results]))
    mean_content_recall = float(np.mean([item["content_stem_recall"] for item in results]))
    mean_audio_rms = float(np.mean([item["audio_rms"] for item in results]))
    mean_latency = float(np.mean(response_latencies)) if response_latencies else None
    max_clipping = float(max(item["audio_clipping_ratio"] for item in results))
    failures = []
    if mean_ukrainian_ratio < args.min_ukrainian_ratio:
        failures.append("mean Ukrainian-letter ratio is below threshold")
    if mean_content_recall < args.min_content_recall:
        failures.append("mean held-out content recall is below threshold")
    if mean_audio_rms < args.min_audio_rms:
        failures.append("mean generated audio RMS is below threshold")
    if max_clipping > args.max_clipping_ratio:
        failures.append("generated audio clipping exceeds threshold")
    if mean_latency is None or mean_latency > args.max_response_latency:
        failures.append("mean response latency is missing or above threshold")
    summary = {
        "cases": len(results),
        "mean_ukrainian_ratio": round(mean_ukrainian_ratio, 4),
        "mean_content_stem_recall": round(mean_content_recall, 4),
        "mean_realtime_factor": round(
            float(np.mean([item["realtime_factor"] for item in results])), 4
        ),
        "mean_response_latency_seconds": (
            round(mean_latency, 4) if mean_latency is not None else None
        ),
        "mean_audio_rms": round(mean_audio_rms, 6),
        "max_clipping_ratio": max_clipping,
        "passed": not failures,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.enforce and failures:
        raise SystemExit("Ukrainian MLX quality gate failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("training/uk_UA/mlx-runs/uk-v3/model.moshi_mlx.safetensors"),
    )
    parser.add_argument("--config", type=Path, default=Path("model-mlx/config.json"))
    parser.add_argument("--mimi", type=Path, default=Path("model-mlx/tokenizer.safetensors"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("training/uk_UA/data-v3-eval/eval.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("training/uk_UA/eval-runs/uk-v3"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--lead-seconds", type=float, default=0.5)
    parser.add_argument("--tail-seconds", type=float, default=6.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--text-top-k", type=int, default=25)
    parser.add_argument("--audio-top-k", type=int, default=250)
    parser.add_argument("--quantized", type=int, choices=(4, 8))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-ukrainian-ratio", type=float, default=0.95)
    parser.add_argument("--min-content-recall", type=float, default=0.25)
    parser.add_argument("--min-audio-rms", type=float, default=0.01)
    parser.add_argument("--max-clipping-ratio", type=float, default=0.0001)
    parser.add_argument("--max-response-latency", type=float, default=3.0)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    args.tokenizer = _resolve_tokenizer(args.tokenizer)
    for path in (args.weights, args.config, args.mimi, args.tokenizer, args.manifest):
        if not path.exists():
            parser.error(f"missing file: {path}")
    if args.limit < 1 or min(args.lead_seconds, args.tail_seconds) < 0:
        parser.error("--limit must be positive and padding durations cannot be negative")
    if not all(
        0.0 <= value <= 1.0
        for value in (
            args.min_ukrainian_ratio,
            args.min_content_recall,
            args.max_clipping_ratio,
        )
    ):
        parser.error("ratio thresholds must be between zero and one")
    if args.min_audio_rms < 0 or args.max_response_latency <= 0:
        parser.error("audio RMS must be non-negative and max latency must be positive")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
