"""Enforces the project's cost model: the server daemon (orchestrator/)
uses Claude Haiku exclusively, never Sonnet or Opus (README Section 2).
Sonnet is a local/dev-time-only tool and must never appear in code that
runs on the 24/7 server -- concretely enforced here, not just documented,
the same way test_safety_isolation.py enforces the safety/LLM separation.
"""

from pathlib import Path

ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1] / "orchestrator"


def test_orchestrator_never_uses_a_sonnet_or_opus_model_id():
    # Checks for the actual model-id prefixes (e.g. "claude-sonnet-5"), not
    # the bare words "sonnet"/"opus" -- those legitimately appear in
    # docstrings/comments explaining this very policy (see
    # status_formatter.py's module docstring), which isn't a violation of it.
    offenders = [
        str(path)
        for path in ORCHESTRATOR_DIR.rglob("*.py")
        if "claude-sonnet" in path.read_text().lower() or "claude-opus" in path.read_text().lower()
    ]
    assert not offenders, f"orchestrator/ must only use Haiku model ids, never Sonnet/Opus: {offenders}"


def test_orchestrator_uses_haiku_model_id():
    formatter = (ORCHESTRATOR_DIR / "status_formatter.py").read_text()
    assert "haiku" in formatter.lower()
