"""Unit tests for llm_pipeline/novel_condition_tester.py's Clause/
ConditionSpec validation -- this whitelist is the entire boundary
between an LLM's proposal and code that actually runs; it must reject
anything outside it at construction time, not fail confusingly later.
"""
from __future__ import annotations

import pytest

from llm_pipeline.novel_condition_tester import SUPPORTED_INDICATORS, Clause, ConditionSpec, condition_desc


def test_every_supported_indicator_constructs_a_valid_clause():
    for indicator in SUPPORTED_INDICATORS:
        clause = Clause(indicator=indicator, op=">", threshold=1.0)
        assert clause.indicator == indicator


def test_unsupported_indicator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported indicator"):
        Clause(indicator="totally_made_up", op=">", threshold=1.0)


def test_unsupported_operator_is_rejected():
    with pytest.raises(ValueError, match="Unsupported operator"):
        Clause(indicator="close_return_1d", op="==", threshold=1.0)


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="direction must be"):
        ConditionSpec(label="x", clauses=(Clause(indicator="close_return_1d", op=">", threshold=1.0),), direction="sideways")


def test_empty_clauses_is_rejected():
    with pytest.raises(ValueError, match="at least one clause"):
        ConditionSpec(label="x", clauses=(), direction="long")


def test_multi_clause_spec_is_anded_in_its_description():
    spec = ConditionSpec(
        label="x",
        clauses=(Clause(indicator="rsi_14d", op="<", threshold=30.0), Clause(indicator="shock_zscore", op=">=", threshold=3.0)),
        direction="long",
    )
    desc = condition_desc(spec)
    assert " AND " in desc
    assert desc.count(" AND ") == 1  # two clauses, joined once -- not silently collapsed to one
    assert "below 30.0" in desc and "at least 3.0" in desc
    # indicator names are translated to plain language, not shown as raw variable names
    assert "rsi_14d" not in desc and "shock_zscore" not in desc
    # comparisons are translated to plain English too -- a raw "<"/">" HTML-escaped
    # to "&lt;"/"&gt;" was observed live to sometimes render as literal text in
    # Telegram instead of decoding back to the symbol; never generating the raw
    # symbol at all sidesteps that failure mode entirely
    assert "<" not in desc and ">" not in desc
