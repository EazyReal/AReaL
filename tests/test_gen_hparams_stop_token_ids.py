# SPDX-License-Identifier: Apache-2.0
"""Regression tests for stop_token_ids construction when pad/eos token id is None.

Base-Llama-style tokenizers have ``pad_token_id is None``; that None must not land in
the int-typed ``stop_token_ids`` list.
"""

import pytest

from areal.api.cli_args import GenerationHyperparameters


class _TokenizerStub:
    def __init__(self, pad_token_id, eos_token_id):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id


class TestStopTokenIdsNoneGuard:
    @pytest.mark.parametrize(
        "pad_token_id, eos_token_id, expected",
        [(None, 2, 2), (0, None, 0)],
    )
    def test_none_token_id_not_injected(self, pad_token_id, eos_token_id, expected):
        """A None pad/eos token id is skipped, not injected into the int-typed list."""
        tokenizer = _TokenizerStub(pad_token_id=pad_token_id, eos_token_id=eos_token_id)

        new_gconfig = GenerationHyperparameters().new_with_stop_and_pad_token_ids(
            tokenizer
        )

        assert None not in new_gconfig.stop_token_ids
        assert expected in new_gconfig.stop_token_ids
        assert all(isinstance(tid, int) for tid in new_gconfig.stop_token_ids)

    def test_valid_ids_added_without_duplicates(self):
        """Valid pad/eos ids are added once and existing ids are preserved."""
        tokenizer = _TokenizerStub(pad_token_id=2, eos_token_id=2)
        gconfig = GenerationHyperparameters(stop_token_ids=[7])

        new_gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)

        assert new_gconfig.stop_token_ids.count(2) == 1
        assert 7 in new_gconfig.stop_token_ids

    def test_list_valued_eos_token_id(self):
        """A list-valued eos_token_id (e.g. Llama 3) has its ids added individually."""
        tokenizer = _TokenizerStub(pad_token_id=0, eos_token_id=[128001, 128009])
        gconfig = GenerationHyperparameters()

        new_gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)

        assert 128001 in new_gconfig.stop_token_ids
        assert 128009 in new_gconfig.stop_token_ids
        assert all(isinstance(tid, int) for tid in new_gconfig.stop_token_ids)
