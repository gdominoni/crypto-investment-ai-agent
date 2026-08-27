"""Unit tests for llm_pipeline/novel_condition_tester.py's ConditionSpec
validation -- this whitelist is the entire boundary between an LLM's
proposal and code that actually runs; it must reject anything outside it
at construction time, not fail confusingly later.
"""
from __future__ import annotations

import pytest

from llm_pipeline.novel_condition_tester import SUPPORTED_INDICATORS, ConditionSpec


def test_every_supported_indicator_constructs_a_valid_spec():
    for indicator in SUPPORTED_INDICATORS:
        spec = ConditionSpec(label="x", indicator=indicator, op=">", threshold=1.0, direction="long")
        assert spec.indicator == indicator


def test_unsupported_indicator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported indicator"):
        ConditionSpec(label="x", indicator="totally_made_up", op=">", threshold=1.0, direction="long")


def test_unsupported_operator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported operator"):
        ConditionSpec(label="x", indicator="close_return_1d", op="==", threshold=1.0, direction="long")


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="direction must be"):
        ConditionSpec(label="x", indicator="close_return_1d", op=">", threshold=1.0, direction="sideways")
