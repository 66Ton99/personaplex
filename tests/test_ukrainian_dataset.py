import json
import random
import tempfile
import unittest
import wave
from pathlib import Path

from scripts.build_common_voice_uk_dataset import (
    _clean_response,
    _echo_response,
    _response_is_grounded,
    _supervised_response,
    _ukrainian_ratio,
    _word_alignments,
)
from scripts.validate_moshi_dataset import _simulate_interleaver, validate_dataset
from scripts.score_ukrainian_audio_intelligibility import _word_error_rate
from scripts.evaluate_ukrainian_mlx import _wrap_system_prompt
from scripts.validate_mlx_tokens import _sample_indices
from moshi.prepare_mlx_data import _build_text_stream


class FakeTokenizer:
    def encode(self, text):
        return [ord(char) for char in text.replace(" ", "")]

    def id_to_piece(self, token):
        return chr(token)


class UkrainianDatasetTest(unittest.TestCase):
    def test_ukrainian_ratio(self):
        self.assertGreater(_ukrainian_ratio("Привіт, як справи?"), 0.9)
        self.assertEqual(_ukrainian_ratio("Hello"), 0.0)

    def test_clean_response_removes_unspoken_symbols(self):
        self.assertEqual(_clean_response("Відповідь: Чудово! ☀️"), "Чудово!")

    def test_echo_response_preserves_user_transcript(self):
        self.assertEqual(_echo_response("Система працює."), "Ви сказали: Система працює.")

    def test_supervised_response_preserves_user_transcript(self):
        response = _supervised_response("Система працює.", random.Random(7))
        self.assertIn("Система працює.", response)

    def test_response_grounding_rejects_unrelated_hallucination(self):
        self.assertTrue(
            _response_is_grounded(
                "Система розділових знаків",
                "Система розділових знаків структурує текст.",
            )
        )
        self.assertFalse(
            _response_is_grounded(
                "Вона слідком бігала.",
                "Можливо, потрібен новий акумулятор.",
            )
        )

    def test_word_error_rate_normalizes_case_and_punctuation(self):
        self.assertEqual(_word_error_rate("Привіт, Україно!", "привіт україно"), 0.0)
        self.assertEqual(_word_error_rate("один два", "один три"), 0.5)

    def test_mlx_validation_samples_cover_manifest(self):
        self.assertEqual(_sample_indices(10, 1), [0])
        self.assertEqual(_sample_indices(10, 3), [0, 4, 9])

    def test_mlx_evaluator_uses_runtime_system_tags(self):
        self.assertEqual(
            _wrap_system_prompt("Відповідай українською."),
            "<system> Відповідай українською. <system>",
        )

    def test_word_alignments_cover_utterance(self):
        alignments = _word_alignments("Вітаю вас!", 1.5, 2.0, FakeTokenizer())
        self.assertLessEqual(alignments[0][1][0], 1.5)
        self.assertEqual(alignments[-1][1][1], 3.5)
        self.assertTrue(all(item[2] == "SPEAKER_MAIN" for item in alignments))

    def test_interleaver_detects_overwritten_tokens(self):
        alignments = [
            ["довгеслово", [0.0, 0.1], "SPEAKER_MAIN"],
            ["ще", [0.08, 0.2], "SPEAKER_MAIN"],
        ]
        tokens, _, dropped, _ = _simulate_interleaver(
            alignments, FakeTokenizer(), duration=0.5
        )
        self.assertEqual(tokens, 12)
        self.assertGreater(dropped, 0)

    def test_mlx_text_stream_preserves_scheduled_tokens(self):
        alignments = [
            ["Так", [0.0, 0.3], "SPEAKER_MAIN"],
            ["добре", [0.3, 0.8], "SPEAKER_MAIN"],
        ]
        stream = _build_text_stream(alignments, FakeTokenizer(), frame_count=16)
        emitted = [token for token in stream if token not in (0, 3)]
        self.assertEqual(emitted, [ord(char) for char in "Такдобре"])

    def test_mlx_text_stream_queues_dense_alignments(self):
        alignments = [
            ["довгеслово", [0.0, 0.1], "SPEAKER_MAIN"],
            ["ще", [0.08, 0.2], "SPEAKER_MAIN"],
        ]
        stream = _build_text_stream(alignments, FakeTokenizer(), frame_count=16)
        emitted = [token for token in stream if token not in (0, 3)]
        self.assertEqual(emitted, [ord(char) for char in "довгесловоще"])

    def test_valid_stereo_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_path = root / "sample.wav"
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(24_000)
                output.writeframes(b"\0" * 24_000 * 2 * 2)
            wav_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "alignments": [
                            ["Так", [0.1, 0.4], "SPEAKER_MAIN"],
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps({"path": str(wav_path), "duration": 1.0}) + "\n",
                encoding="utf-8",
            )

            report = validate_dataset(manifest, FakeTokenizer())

            self.assertEqual(report.errors, [])
            self.assertEqual(report.files, 1)


if __name__ == "__main__":
    unittest.main()
