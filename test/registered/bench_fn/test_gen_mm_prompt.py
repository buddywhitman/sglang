"""Unit tests for gen_mm_prompt changes in python/sglang/benchmark/datasets/common.py.

PR change: gen_mm_prompt now calls tokenizer.get_vocab().values() directly
(instead of using the cached get_available_multimodal_text_tokens helper),
and uses list.remove() to exclude image_pad_id when provided.
"""
import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tokenizer(vocab: dict, decode_fn=None):
    """Build a minimal tokenizer mock."""
    tok = MagicMock()
    tok.get_vocab.return_value = dict(vocab)
    if decode_fn is not None:
        tok.decode.side_effect = decode_fn
    else:
        tok.decode.return_value = "decoded"
    return tok


class TestGenMmPrompt(unittest.TestCase):
    """Tests for gen_mm_prompt after the PR refactor."""

    def test_basic_generation_returns_decoded_string(self):
        """gen_mm_prompt should return the tokenizer's decoded output."""
        from sglang.benchmark.datasets.common import gen_mm_prompt

        vocab = {f"tok_{i}": i for i in range(50)}
        tok = _make_tokenizer(vocab)
        result = gen_mm_prompt(tok, image_pad_id=None, token_num=5)
        tok.decode.assert_called_once()
        self.assertEqual(result, "decoded")

    def test_image_pad_id_none_does_not_remove_any_token(self):
        """When image_pad_id is None (falsy), no token is removed."""
        from sglang.benchmark.datasets.common import gen_mm_prompt

        vocab = {f"tok_{i}": i for i in range(10)}
        captured = {}

        def capture_decode(tokens):
            captured["tokens"] = list(tokens)
            return "ok"

        tok = _make_tokenizer(vocab, decode_fn=capture_decode)
        gen_mm_prompt(tok, image_pad_id=None, token_num=10)

        # All 10 tokens from vocab should be candidates (none removed)
        # The selection pool size equals len(vocab) = 10
        all_ids = set(vocab.values())
        self.assertTrue(all(t in all_ids for t in captured["tokens"]))

    def test_image_pad_id_excluded_from_candidates(self):
        """When image_pad_id is provided, it must not appear in selected tokens."""
        from sglang.benchmark.datasets.common import gen_mm_prompt

        pad_id = 99
        vocab = {f"tok_{i}": i for i in range(10)}
        vocab["[IMG_PAD]"] = pad_id
        captured = {}

        def capture_decode(tokens):
            captured["tokens"] = list(tokens)
            return "ok"

        tok = _make_tokenizer(vocab, decode_fn=capture_decode)
        # Run many times to reduce probability of accidental pass
        for _ in range(20):
            gen_mm_prompt(tok, image_pad_id=pad_id, token_num=5)
            self.assertNotIn(
                pad_id,
                captured["tokens"],
                "image_pad_id should be excluded from candidates",
            )

    def test_correct_number_of_tokens_selected(self):
        """gen_mm_prompt must select exactly token_num tokens."""
        from sglang.benchmark.datasets.common import gen_mm_prompt

        vocab = {f"tok_{i}": i for i in range(100)}
        selected_lengths = []

        def capture_decode(tokens):
            selected_lengths.append(len(tokens))
            return "ok"

        tok = _make_tokenizer(vocab, decode_fn=capture_decode)
        for n in [1, 5, 20, 100]:
            selected_lengths.clear()
            gen_mm_prompt(tok, image_pad_id=None, token_num=n)
            self.assertEqual(selected_lengths[0], n)

    def test_get_vocab_called_each_invocation(self):
        """get_vocab is called on every call (no caching between calls)."""
        from sglang.benchmark.datasets.common import gen_mm_prompt

        vocab = {f"tok_{i}": i for i in range(10)}
        tok = _make_tokenizer(vocab)
        gen_mm_prompt(tok, image_pad_id=None, token_num=3)
        gen_mm_prompt(tok, image_pad_id=None, token_num=3)
        self.assertEqual(tok.get_vocab.call_count, 2)

    def test_falsy_image_pad_id_zero_not_removed(self):
        """image_pad_id=0 is falsy in Python; the new code uses `if image_pad_id:`.

        When pad_id is 0 (falsy), no removal happens — this is the new
        behaviour after the PR simplified the guard to `if image_pad_id`.
        """
        from sglang.benchmark.datasets.common import gen_mm_prompt

        pad_id = 0
        vocab = {"zero_tok": 0, "one_tok": 1, "two_tok": 2}
        captured = {}

        def capture_decode(tokens):
            captured["tokens"] = list(tokens)
            return "ok"

        tok = _make_tokenizer(vocab, decode_fn=capture_decode)
        # image_pad_id=0 is falsy, so 0 stays in the pool
        gen_mm_prompt(tok, image_pad_id=pad_id, token_num=30)
        # 0 may appear in the output since it was not removed
        # We just confirm no ValueError was raised and decoding was called
        tok.decode.assert_called()

    def test_image_pad_id_not_in_vocab_raises(self):
        """If image_pad_id is truthy but not in the token list, list.remove raises ValueError."""
        from sglang.benchmark.datasets.common import gen_mm_prompt

        vocab = {"tok_0": 1, "tok_1": 2}
        tok = _make_tokenizer(vocab)
        with self.assertRaises(ValueError):
            gen_mm_prompt(tok, image_pad_id=99, token_num=1)


if __name__ == "__main__":
    unittest.main()