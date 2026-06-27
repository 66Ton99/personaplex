#!/usr/bin/env python3
"""Build Moshi fine-tuning samples from Ukrainian Common Voice clips.

Common Voice audio is placed on the user channel. A local OpenAI-compatible
model writes a short Ukrainian response, and macOS ``say`` renders that response
on the assistant channel. The resulting WAV keeps Moshi's channel convention:
assistant on channel 0, user on channel 1.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece
import sphn


SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
CYRILLIC_RE = re.compile(r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]")
WORD_RE = re.compile(r"\S+")
SPOKEN_PUNCTUATION = frozenset(".,!?;:…'’\"-—()")
UKRAINIAN_STOPWORDS = frozenset(
    {
        "але", "вона", "вони", "воно", "його", "йому", "коли", "мене",
        "може", "нього", "після", "перед", "сама", "себе", "також", "тому",
        "хтось", "через", "щось", "який", "яка", "яке", "якщо",
    }
)


@dataclass(frozen=True)
class CommonVoiceRow:
    clip_path: Path
    sentence: str


@dataclass(frozen=True)
class ExchangePlan:
    index: int
    row: CommonVoiceRow
    use_echo: bool
    assistant_text: str | None
    overlap: float
    interruption: float


@dataclass(frozen=True)
class PreparedExchange:
    plan: ExchangePlan
    assistant_text: str
    user_audio: np.ndarray
    assistant_audio: np.ndarray
    generated_response: bool


def _ukrainian_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(CYRILLIC_RE.fullmatch(char)) for char in letters) / len(letters)


def _load_rows(tsv_path: Path, clips_dir: Path) -> list[CommonVoiceRow]:
    rows: list[CommonVoiceRow] = []
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            sentence = (raw.get("sentence") or "").strip()
            relative_path = (raw.get("path") or "").strip()
            if not sentence or not relative_path or _ukrainian_ratio(sentence) < 0.8:
                continue
            clip_path = clips_dir / relative_path
            if clip_path.exists():
                rows.append(CommonVoiceRow(clip_path=clip_path, sentence=sentence))
    return rows


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


def _clean_response(text: str) -> str:
    text = re.sub(r"^(відповідь|response)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip(" \t\r\n\"'«»")
    text = "".join(
        char
        for char in text
        if char.isalnum() or char.isspace() or char in SPOKEN_PUNCTUATION
    )
    return re.sub(r"\s+", " ", text).strip()


def _content_stems(text: str) -> set[str]:
    words = re.findall(r"[А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]{4,}", text.lower())
    return {word[:5] for word in words if word not in UKRAINIAN_STOPWORDS}


def _response_is_grounded(user_text: str, response: str) -> bool:
    user_stems = _content_stems(user_text)
    return not user_stems or bool(user_stems & _content_stems(response))


def _request_response(endpoint: str, model: str, user_text: str, timeout: float) -> str:
    payload = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 120,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ти створюєш навчальні репліки для українського голосового асистента. "
                    "Відповідай природною сучасною українською мовою одним коротким реченням. "
                    "Обов'язково повтори у відповіді хоча б одне ключове слово з репліки користувача. "
                    "Завжди лише коротко перефразовуй почуте, навіть якщо це питання; "
                    "не відповідай по суті й не додавай причин, фактів, назв або чисел, яких не було. "
                    "Перед відповіддю перевір граматику, узгодження слів і українські відмінки. "
                    "Не додавай пояснень, лапок, списків, емодзі чи службового тексту."
                ),
            },
            {"role": "user", "content": user_text},
        ],
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer lm-studio",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LM Studio rejected the request at {request.full_url} "
            f"with HTTP {error.code}: {details}"
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"LM Studio request failed at {request.full_url}: {error}. "
            "Start the local server and load openai/gpt-oss-20b."
        ) from error

    text = _clean_response(result["choices"][0]["message"]["content"])
    if not text or _ukrainian_ratio(text) < 0.7 or not _response_is_grounded(user_text, text):
        raise RuntimeError(f"Model returned an unusable response: {text!r}")
    return text


def _request_response_resilient(
    endpoint: str, model: str, user_text: str, timeout: float, retries: int = 4
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return _request_response(endpoint, model, user_text, timeout)
        except (RuntimeError, KeyError, IndexError, TypeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    # Keep a rare transient LM failure from discarding an otherwise valid clip.
    if last_error is not None:
        return _echo_response(user_text)
    raise RuntimeError("No response attempts were made")


def _load_response_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    cache: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            cache[item["clip_path"]] = _clean_response(item["assistant_text"])
    return cache


def _append_response_cache(path: Path, clip_path: Path, user_text: str, assistant_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(
            {
                "clip_path": str(clip_path.resolve()),
                "user_text": user_text,
                "assistant_text": assistant_text,
            },
            handle,
            ensure_ascii=False,
        )
        handle.write("\n")


def _read_mono(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, source_rate = sphn.read(str(path))
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeError(f"No mono audio decoded from {path}")
    if source_rate != sample_rate:
        source_x = np.arange(audio.size, dtype=np.float64)
        target_size = max(1, round(audio.size * sample_rate / source_rate))
        target_x = np.linspace(0, audio.size - 1, target_size, dtype=np.float64)
        audio = np.interp(target_x, source_x, audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio


def _synthesize_say(text: str, voice: str, rate: int, output_path: Path) -> np.ndarray:
    command = [
        "say",
        "-v",
        voice,
        "-r",
        str(rate),
        "--data-format=LEI16@24000",
        "-o",
        str(output_path),
        text,
    ]
    subprocess.run(command, check=True)
    return _read_mono(output_path)


def _synthesize_for_token_rate(
    text: str,
    voice: str,
    rate: int,
    output_path: Path,
    tokenizer: sentencepiece.SentencePieceProcessor,
) -> np.ndarray:
    token_count = sum(len(tokenizer.encode(word)) for word in WORD_RE.findall(text))
    minimum_duration = (token_count + 1) / FRAME_RATE
    current_rate = rate
    for _ in range(3):
        audio = _synthesize_say(text, voice, current_rate, output_path)
        duration = audio.size / SAMPLE_RATE
        if duration >= minimum_duration:
            return audio
        current_rate = max(80, int(current_rate * duration / minimum_duration * 0.9))
    # Very short Ukrainian utterances can hit a hard floor in macOS ``say``.
    # Preserve their text timing by padding the rendered audio rather than
    # dropping an otherwise valid Common Voice exchange.
    target_samples = math.ceil(minimum_duration * SAMPLE_RATE)
    if audio.size < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.size))
    return audio


def _load_silero_tts() -> object:
    import torch

    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ua",
        speaker="v4_ua",
        trust_repo=True,
    )
    return model


def _synthesize_silero_for_token_rate(
    text: str,
    speaker: str,
    model: object,
    tokenizer: sentencepiece.SentencePieceProcessor,
) -> np.ndarray:
    import torch

    with torch.inference_mode():
        rendered = model.apply_tts(  # type: ignore[attr-defined]
            text=text,
            speaker=speaker,
            sample_rate=SAMPLE_RATE,
        )
    audio = np.asarray(rendered.detach().cpu(), dtype=np.float32).reshape(-1)
    token_count = sum(len(tokenizer.encode(word)) for word in WORD_RE.findall(text))
    target_samples = math.ceil((token_count + 1) / FRAME_RATE * SAMPLE_RATE)
    if audio.size < target_samples:
        audio = np.pad(audio, (0, target_samples - audio.size))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return audio


def _echo_response(text: str) -> str:
    return _clean_response(f"Ви сказали: {text}")


def _supervised_response(text: str, rng: random.Random) -> str:
    template = rng.choice(
        (
            "Ви сказали: {text}",
            "Я почув вас: {text}",
            "Правильно розумію: {text}",
            "Повторю почуте: {text}",
        )
    )
    return _clean_response(template.format(text=text))


def _word_alignments(
    text: str,
    start_sec: float,
    duration_sec: float,
    tokenizer: sentencepiece.SentencePieceProcessor,
) -> list[list[object]]:
    words = WORD_RE.findall(text)
    if not words:
        return []
    token_counts = [len(tokenizer.encode(word)) for word in words]
    total_tokens = sum(token_counts)
    if total_tokens / FRAME_RATE > duration_sec:
        raise RuntimeError(
            f"{total_tokens} tokens do not fit into {duration_sec:.2f}s of assistant audio"
        )
    base_frame = math.floor(start_sec * FRAME_RATE)
    token_cursor = 0
    alignments: list[list[object]] = []
    for index, (word, token_count) in enumerate(zip(words, token_counts)):
        word_start = (base_frame + token_cursor) / FRAME_RATE
        token_cursor += token_count
        end = (
            start_sec + duration_sec
            if index == len(words) - 1
            else (base_frame + token_cursor) / FRAME_RATE
        )
        alignments.append([word, [round(word_start, 4), round(end, 4)], "SPEAKER_MAIN"])
    return alignments


def _write_stereo(
    path: Path,
    segments: list[tuple[int, float, np.ndarray]],
    minimum_duration: float,
) -> float:
    frame_count = max(
        round(minimum_duration * SAMPLE_RATE),
        *(round(start * SAMPLE_RATE) + audio.size for _, start, audio in segments),
    )
    stereo = np.zeros((frame_count, 2), dtype=np.float32)
    for channel, start, audio in segments:
        offset = round(start * SAMPLE_RATE)
        stereo[offset : offset + audio.size, channel] += audio
    stereo = np.clip(stereo, -1.0, 1.0)
    pcm = (stereo * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())
    return frame_count / SAMPLE_RATE


def _iter_selected(rows: list[CommonVoiceRow], limit: int, seed: int) -> Iterable[CommonVoiceRow]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    yield from shuffled[:limit]


def _plan_exchange(
    index: int,
    row: CommonVoiceRow,
    args: argparse.Namespace,
    rng: random.Random,
    cache: dict[str, str],
) -> ExchangePlan:
    use_echo = rng.random() < args.echo_probability
    if use_echo:
        assistant_text = _supervised_response(row.sentence, rng)
    else:
        assistant_text = cache.get(str(row.clip_path.resolve()))
        if assistant_text is not None and not _response_is_grounded(
            row.sentence, assistant_text
        ):
            assistant_text = None
    overlap = (
        rng.uniform(0.0, args.max_overlap)
        if rng.random() < args.overlap_probability
        else 0.0
    )
    interruption = (
        rng.uniform(0.0, args.max_interruption)
        if rng.random() < args.interruption_probability
        else 0.0
    )
    return ExchangePlan(
        index=index,
        row=row,
        use_echo=use_echo,
        assistant_text=assistant_text,
        overlap=overlap,
        interruption=interruption,
    )


def _prepare_exchange(
    plan: ExchangePlan,
    args: argparse.Namespace,
    tokenizer: sentencepiece.SentencePieceProcessor,
    silero_model: object | None,
    temp_dir: Path,
) -> PreparedExchange:
    assistant_text = plan.assistant_text
    generated_response = assistant_text is None
    if assistant_text is None:
        assistant_text = _request_response_resilient(
            args.lm_studio_url,
            args.model,
            plan.row.sentence,
            args.timeout,
        )

    user_audio = _read_mono(plan.row.clip_path)
    if silero_model is None:
        assistant_audio = _synthesize_for_token_rate(
            assistant_text,
            args.voice,
            args.speech_rate,
            temp_dir / f"assistant-{plan.index:06d}.wav",
            tokenizer,
        )
    else:
        assistant_audio = _synthesize_silero_for_token_rate(
            assistant_text,
            args.silero_speaker,
            silero_model,
            tokenizer,
        )
    return PreparedExchange(
        plan=plan,
        assistant_text=assistant_text,
        user_audio=user_audio,
        assistant_audio=assistant_audio,
        generated_response=generated_response,
    )


def _load_completed_clips(manifest_path: Path) -> tuple[list[dict[str, object]], set[str]]:
    items: list[dict[str, object]] = []
    clips: set[str] = set()
    if not manifest_path.exists():
        return items, clips
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            items.append(item)
            metadata_path = Path(str(item["path"])).with_suffix(".json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for exchange in metadata.get("source", {}).get("exchanges", []):
                clip_path = exchange.get("clip_path")
                if clip_path:
                    clips.add(str(clip_path))
    return items, clips


def build_dataset(args: argparse.Namespace) -> None:
    tsv_path = args.common_voice_root / args.tsv
    clips_dir = args.common_voice_root / "clips"
    rows = _load_rows(tsv_path, clips_dir)
    if not rows:
        raise SystemExit(f"No usable Ukrainian rows found in {tsv_path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.split}.jsonl"
    if manifest_path.exists() and not args.resume:
        manifest_path.unlink()
    cache_path = output_dir / "responses.jsonl"
    cache = _load_response_cache(cache_path)
    rng = random.Random(args.seed)
    tokenizer_path = _resolve_tokenizer(args.tokenizer)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
    silero_model = _load_silero_tts() if args.tts_backend == "silero" else None

    manifest_items, completed_clips = _load_completed_clips(manifest_path)
    session_segments: list[tuple[int, float, np.ndarray]] = []
    session_alignments: list[list[object]] = []
    session_sources: list[dict[str, str]] = []
    session_cursor = args.lead_in
    session_index = len(manifest_items)

    def flush_session() -> None:
        nonlocal session_cursor, session_index
        if not session_segments:
            return
        stem = f"{args.split}-{session_index:06d}"
        wav_path = output_dir / f"{stem}.wav"
        duration = _write_stereo(
            wav_path,
            session_segments,
            minimum_duration=args.session_duration,
        )
        metadata = {
            "alignments": session_alignments,
            "source": {
                "dataset": "Mozilla Common Voice",
                "exchanges": session_sources,
            },
        }
        wav_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        item = {"path": str(wav_path), "duration": round(duration, 4)}
        manifest_items.append(item)
        with manifest_path.open("a", encoding="utf-8") as manifest:
            json.dump(item, manifest)
            manifest.write("\n")
        session_segments.clear()
        session_alignments.clear()
        session_sources.clear()
        session_cursor = args.lead_in
        session_index += 1

    with tempfile.TemporaryDirectory(prefix="personaplex-uk-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        selected = [
            row
            for row in _iter_selected(rows, args.limit, args.seed)
            if str(row.clip_path.resolve()) not in completed_clips
        ]
        plans: Iterable[ExchangePlan] = (
            _plan_exchange(index, row, args, rng, cache)
            for index, row in enumerate(selected)
        )

        plan_iter = iter(plans)
        pending: deque[Future[PreparedExchange]] = deque()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(args.workers * 2):
                try:
                    plan = next(plan_iter)
                except StopIteration:
                    break
                pending.append(
                    executor.submit(
                        _prepare_exchange,
                        plan,
                        args,
                        tokenizer,
                        silero_model,
                        temp_dir,
                    )
                )

            while pending:
                prepared = pending.popleft().result()
                plan = prepared.plan
                row = plan.row
                assistant_text = prepared.assistant_text
                user_audio = prepared.user_audio
                assistant_audio = prepared.assistant_audio
                cache_key = str(row.clip_path.resolve())
                if prepared.generated_response:
                    cache[cache_key] = assistant_text
                    _append_response_cache(
                        cache_path,
                        row.clip_path,
                        row.sentence,
                        assistant_text,
                    )

                try:
                    next_plan = next(plan_iter)
                except StopIteration:
                    pass
                else:
                    pending.append(
                        executor.submit(
                            _prepare_exchange,
                            next_plan,
                            args,
                            tokenizer,
                            silero_model,
                            temp_dir,
                        )
                    )

                user_start = session_cursor
                user_end = user_start + user_audio.size / SAMPLE_RATE
                overlap = plan.overlap
                assistant_start = max(user_start, user_end + args.response_gap - overlap)
                assistant_end = assistant_start + assistant_audio.size / SAMPLE_RATE
                exchange_end = max(user_end, assistant_end)

                if session_segments and exchange_end + args.tail > args.session_duration:
                    flush_session()
                    user_start = session_cursor
                    user_end = user_start + user_audio.size / SAMPLE_RATE
                    assistant_start = max(user_start, user_end + args.response_gap - overlap)
                    assistant_end = assistant_start + assistant_audio.size / SAMPLE_RATE
                    exchange_end = max(user_end, assistant_end)

                session_segments.extend(
                    [
                        (0, assistant_start, assistant_audio),
                        (1, user_start, user_audio),
                    ]
                )
                session_alignments.extend(
                    _word_alignments(
                        assistant_text,
                        assistant_start,
                        assistant_audio.size / SAMPLE_RATE,
                        tokenizer,
                    )
                )
                session_sources.append(
                    {
                        "clip_path": cache_key,
                        "user_text": row.sentence,
                        "assistant_text": assistant_text,
                        "response_kind": "echo" if plan.use_echo else "natural",
                        "tts_backend": args.tts_backend,
                    }
                )

                session_cursor = exchange_end + args.tail
                if plan.interruption:
                    session_cursor = max(user_end, assistant_end - plan.interruption)
                print(f"[{plan.index + 1}/{len(selected)}] prepared {row.clip_path.name}")

        flush_session()

    print(f"Wrote {len(manifest_items)} samples to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("common_voice_root", type=Path, help="Extracted Ukrainian Common Voice directory")
    parser.add_argument("--tsv", default="train.tsv", help="TSV file under the Common Voice root")
    parser.add_argument("--output-dir", type=Path, default=Path("training/uk_UA/data"))
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lm-studio-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--voice", default="Lesya", help="macOS Ukrainian say voice")
    parser.add_argument("--speech-rate", type=int, default=185)
    parser.add_argument("--tts-backend", choices=("say", "silero"), default="silero")
    parser.add_argument("--silero-speaker", default="mykyta")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--echo-probability",
        type=float,
        default=0.7,
        help="Fraction of exchanges that explicitly repeat the user transcript.",
    )
    parser.add_argument("--lead-in", type=float, default=0.5)
    parser.add_argument("--response-gap", type=float, default=0.25)
    parser.add_argument("--overlap-probability", type=float, default=0.15)
    parser.add_argument("--max-overlap", type=float, default=0.35)
    parser.add_argument("--interruption-probability", type=float, default=0.15)
    parser.add_argument("--max-interruption", type=float, default=0.5)
    parser.add_argument("--tail", type=float, default=0.5, help="Gap between exchanges")
    parser.add_argument("--session-duration", type=float, default=80.0)
    args = parser.parse_args()
    if args.limit < 1 or args.workers < 1:
        parser.error("--limit and --workers must be positive")
    if not 0.0 <= args.overlap_probability <= 1.0:
        parser.error("--overlap-probability must be between 0 and 1")
    if not 0.0 <= args.interruption_probability <= 1.0:
        parser.error("--interruption-probability must be between 0 and 1")
    if not 0.0 <= args.echo_probability <= 1.0:
        parser.error("--echo-probability must be between 0 and 1")
    if args.session_duration <= 0 or args.speech_rate <= 0:
        parser.error("--session-duration and --speech-rate must be positive")
    if min(
        args.max_overlap,
        args.max_interruption,
        args.lead_in,
        args.response_gap,
        args.tail,
    ) < 0:
        parser.error("timing values cannot be negative")
    return args


if __name__ == "__main__":
    build_dataset(parse_args())
