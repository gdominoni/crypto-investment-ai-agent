"""Syntax this project uses must parse on the oldest Python it claims to support.

CI runs 3.11 and the README advertises "Python 3.11+". A developer on 3.12 or
newer cannot discover a violation by running the tests, because their own
interpreter accepts it -- which is exactly what happened: an f-string nesting a
double quote inside a double-quoted f-string (PEP 701, legal from 3.12) sat in
`format_compression_message` from commit b32683d, and every CI run since then
failed at COLLECTION with

    SyntaxError: unterminated string literal

That is worse than a failing test. It is a syntax error at import, so pytest
could not collect `tests/test_replay_halt.py` at all and the entire suite
stopped -- 200 tests never ran, and the failure looked like a problem with the
replay-halt tests rather than with a message formatter they happen to import.

This test therefore does textually what the local interpreter will not do.
"""
from __future__ import annotations

import pathlib
import re

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Directories that are not this project's code: third-party sources routinely
# require a newer Python than we do, and are not ours to fix.
SKIP_PARTS = {".git", ".venv", "venv", "site-packages", "freqtrade_userdir",
              "__pycache__", "node_modules"}


def _nested_same_quote_fstrings(source: str) -> list[tuple[int, str]]:
    """Lines with an f-string that nests its OWN quote character inside {...}.

    Scans textually and not with `ast`: `ast.parse(..., feature_version=(3, 11))`
    does NOT reject this, because PEP 701 changed the tokenizer rather than the
    grammar. Verified against a known-bad snippet in the test below, so this
    helper cannot silently degrade into one that finds nothing."""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        for match in re.finditer(r"""(\bf|\brf|\bfr)(["'])""", line):
            quote = match.group(2)
            i, depth = match.end(), 0
            while i < len(line):
                ch = line[i]
                if ch == "\\":
                    i += 2
                    continue
                if ch == "{":
                    if i + 1 < len(line) and line[i + 1] == "{":
                        i += 2
                        continue
                    depth += 1
                elif ch == "}":
                    depth = max(0, depth - 1)
                elif ch == quote:
                    if depth > 0:
                        hits.append((lineno, line.strip()))
                    break
                i += 1
    return hits


def _project_python_files() -> list[pathlib.Path]:
    return [p for p in sorted(PROJECT_ROOT.rglob("*.py"))
            if not SKIP_PARTS & set(p.parts)]


def test_the_detector_catches_a_known_violation():
    """Guards the guard. A scanner that quietly matches nothing would let this
    whole file pass forever while CI kept failing."""
    # Assembled at runtime rather than written as a literal: a literal would be
    # picked up by this module's own scan of the project, which includes the
    # test suite, and the guard would fail on its own fixture.
    q = chr(34)
    bad = f"x = f{q}{{'a' if p else {q}b{q}}}{q}"
    assert _nested_same_quote_fstrings(bad), "the detector no longer detects anything"
    good = f"x = f{q}{{'a' if p else 'b'}}{q}"
    assert not _nested_same_quote_fstrings(good), "the detector flags valid code"


def test_no_source_file_needs_a_python_newer_than_ci_runs():
    offenders = []
    for path in _project_python_files():
        for lineno, text in _nested_same_quote_fstrings(path.read_text(errors="ignore")):
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{lineno}: {text[:100]}")
    assert not offenders, (
        "f-string nests its own quote character -- valid on 3.12+, a SyntaxError on "
        "3.11 which is what CI runs and what the README promises. Bind the value to a "
        "local first.\n  " + "\n  ".join(offenders))


def test_every_source_file_actually_parses():
    """Cheap backstop for syntax errors this file's own scanner doesn't model."""
    import ast

    broken = []
    for path in _project_python_files():
        try:
            ast.parse(path.read_text(errors="ignore"))
        except SyntaxError as e:
            broken.append(f"{path.relative_to(PROJECT_ROOT)}:{e.lineno}: {e.msg}")
    assert not broken, "files do not parse:\n  " + "\n  ".join(broken)


# The heavy dependencies CI deliberately does NOT install. `.github/workflows/
# tests.yml` installs a deliberate subset and says why: rather than install an
# exchange client so a string-chunking test can import telegram/bot.py, the one
# call site that needs it imports it lazily. Its own note ends "If a future test
# needs a heavy dependency, prefer making that import lazy over adding it here."
UNINSTALLED_ON_CI = ["ccxt", "freqtrade"]

# Modules a test may import. Each must be importable WITHOUT the packages above:
# an eager import anywhere in their chain makes an exchange client a hard
# requirement of running the tests at all.
MUST_IMPORT_WITHOUT_HEAVY_DEPS = [
    "replay.engine", "replay.orchestrator", "replay.judgment",
    "scheduler.live_daemon", "scheduler.weekly_revalidation",
    "telegram.bot", "llm_pipeline.haiku_sonnet_pipeline",
    "data_ingestion.market_data.binance_fetcher",
]


@pytest.mark.parametrize("module", MUST_IMPORT_WITHOUT_HEAVY_DEPS)
def test_it_imports_without_the_packages_ci_omits(module):
    """Run in a subprocess with those packages blocked at the import hook.

    This failed for real: binance_fetcher imported ccxt at module level,
    weekly_revalidation imports binance_fetcher, live_daemon imports that -- so
    three tests that only wanted to read live_daemon's SOURCE could not run on
    CI. A developer with the full requirements installed cannot reproduce it,
    which is the same blind spot as the 3.11/3.12 syntax difference above."""
    import subprocess
    import sys
    import textwrap

    blocker = textwrap.dedent(f"""
        import sys
        from importlib.abc import MetaPathFinder
        BLOCKED = {UNINSTALLED_ON_CI!r}
        class _Blocker(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".")[0]
                if root in BLOCKED:
                    raise ImportError(f"No module named {{fullname!r}}")
                return None
        sys.meta_path.insert(0, _Blocker())
        try:
            import {UNINSTALLED_ON_CI[0]}
        except ImportError:
            pass
        else:
            raise SystemExit("blocker inactive -- this test would prove nothing")
        import {module}
    """)
    proc = subprocess.run([sys.executable, "-c", blocker], capture_output=True,
                          text=True, cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0, (
        f"{module} cannot be imported without {UNINSTALLED_ON_CI}, so it cannot be "
        f"imported on CI. Make the heavy import lazy (inside the function that uses "
        f"it) rather than adding the package to the workflow.\n{proc.stderr.strip()}")
