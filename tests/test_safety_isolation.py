"""Enforces that the safety kernel never imports the Anthropic SDK.

This is the concrete, testable form of the project's "the LLM can never
disable stop-losses or raise leverage" guarantee: not an access-control
check at runtime, but a structural fact -- the safety package has no code
path that can even reach an LLM client, let alone be swayed by one.
"""

from pathlib import Path

SAFETY_DIR = Path(__file__).resolve().parents[1] / "safety"


def test_safety_kernel_never_imports_anthropic():
    offenders = [
        str(path)
        for path in SAFETY_DIR.rglob("*.py")
        if "anthropic" in path.read_text().lower()
    ]
    assert not offenders, f"safety/ must never reference the Anthropic SDK: {offenders}"
