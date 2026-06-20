# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the GSM8K SFT loss-mask boundary.

Byte-level/BPE tokenizers can merge a token across the question/answer join when
the question has no trailing whitespace, so the naive ``len(encode(question))``
boundary is wrong. A deterministic stub tokenizer reproduces that merge without
any downloads.
"""

from datasets import Dataset

import areal.dataset.gsm8k as gsm8k_mod
from areal.dataset.gsm8k import get_gsm8k_sft_dataset


class _ByteMergeTokenizer:
    """Stub tokenizer that greedily merges the pair ``"ow"`` into one token, so
    ``encode("...o" + "w...")`` differs from ``encode("...o")`` at the boundary."""

    eos_token = "<eos>"

    def __init__(self):
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            step = 2 if text[i : i + 2] == "ow" else 1
            ids.append(self._vocab.setdefault(text[i : i + step], len(self._vocab)))
            i += step
        return ids


def _loss_mask(tokenizer, question: str, answer: str):
    sample = Dataset.from_dict({"question": [question], "answer": [answer]})
    orig, gsm8k_mod.load_dataset = gsm8k_mod.load_dataset, lambda *a, **k: sample
    try:
        dataset = get_gsm8k_sft_dataset(
            path="ignored", split="train", tokenizer=tokenizer
        )
    finally:
        gsm8k_mod.load_dataset = orig
    row = dataset[0]
    return row["input_ids"], row["loss_mask"]


def test_boundary_merge_token_is_supervised():
    """A token spanning the question/answer join is attributed to the answer."""
    tok = _ByteMergeTokenizer()
    question, answer = "abco", "wxyz"  # 'o' + 'w' merge into 'ow' across the join

    prompt_ids = tok.encode(question)
    full_ids = tok.encode(question + answer + tok.eos_token)
    assert full_ids[: len(prompt_ids)] != prompt_ids  # the merge actually happens

    input_ids, loss_mask = _loss_mask(tok, question, answer)
    # Only the 3 clean prompt tokens ('a','b','c') are masked; the merged 'ow'
    # token is supervised, and the masked span is a true prefix of input_ids.
    assert loss_mask == [0] * 3 + [1] * (len(input_ids) - 3)
    assert input_ids[:3] == prompt_ids[:3]


def test_no_merge_boundary_unchanged():
    """With a clean boundary the mask still ends at len(encode(question))."""
    tok = _ByteMergeTokenizer()
    question, answer = "abc", "xyz"

    prompt_ids = tok.encode(question)
    full_ids = tok.encode(question + answer + tok.eos_token)
    assert full_ids[: len(prompt_ids)] == prompt_ids

    _, loss_mask = _loss_mask(tok, question, answer)
    assert loss_mask.count(0) == len(prompt_ids)
