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


def _fstrings_needing_312(source: str) -> list[tuple[int, str]]:
    """Every f-string replacement field that requires Python 3.12 or newer.

    Reads the AST rather than pattern-matching the text, and that difference is
    the whole point of this rewrite. The first version of this guard scanned for
    ONE symptom -- an f-string reusing its own quote character on a single line
    -- and a second violation of the same PEP walked straight past it: an
    expression split across two lines inside an f-string, which is equally legal
    on 3.12 and equally a SyntaxError on 3.11. Enumerating symptoms is a losing
    game; PEP 701 relaxed a category, so the check has to cover the category.

    What it inspects is the EXPRESSION inside each pair of braces, via
    `ast.get_source_segment`, for the three things 3.11 forbids there: a newline,
    a backslash, or the quote character the f-string itself was opened with.

    Note `ast.parse` alone is no help: it accepts all of this on a 3.12+
    interpreter, which is exactly the blind spot being closed.

    This is a fast PRE-FILTER, not the authority. `scripts/check_py311.sh`
    compiles the whole tree with the real Python 3.11 in Docker, pinned to the
    same image CI uses, and that is what settles it when the two disagree -- a
    hand-written check knows the violations someone thought of, the compiler
    knows all of them. This one earns its place by needing no Docker and running
    in the ordinary suite."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # test_every_source_file_actually_parses reports this instead
    lines = source.splitlines()
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        # The delimiter is read off THIS f-string's own source, not guessed from
        # the line it sits on: a single line can hold both f" and f' (there is a
        # real one in forecast/), and guessing flagged every valid inner quote as
        # a violation. A false positive here is worse than a miss -- it is how a
        # guard ends up disabled.
        whole = ast.get_source_segment(source, node) or ""
        m = re.match(r'[rRfFbB]*("""|\'\'\'|"|\')', whole)
        delimiter = m.group(1) if m else ""
        for part in node.values:
            if not isinstance(part, ast.FormattedValue):
                continue
            seg = ast.get_source_segment(source, part.value)
            if seg is None:
                continue
            if "\n" in seg and len(delimiter) == 1:
                # Only for a single-quoted f-string. A triple-quoted one may
                # legitimately carry newlines, and flagging those would be the
                # kind of false positive that gets a guard switched off.
                out.append((part.lineno, f"expression spans multiple lines: {seg.splitlines()[0]!r}"))
                continue
            if "\\" in seg:
                out.append((part.lineno, f"backslash inside the expression: {seg!r}"))
                continue
            if delimiter and len(delimiter) == 1 and delimiter in seg:
                out.append((part.lineno, f"reuses its own {delimiter} delimiter: {seg!r}"))
    return out


def _rejected_on_311(source: str) -> bool:
    """Whether this source would be refused by Python 3.11, however that shows up.

    Two mechanisms, and which one fires depends on the interpreter running the
    tests. On 3.11 the parser itself rejects it. On 3.12+ it parses cleanly and
    only `_fstrings_needing_312` can tell. Callers want the question answered,
    not the mechanism."""
    import ast

    try:
        ast.parse(source)
    except SyntaxError:
        return True
    return bool(_fstrings_needing_312(source))


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
    # Both real violations this project has actually shipped, assembled at
    # runtime so this file does not trip its own scan of the project.
    nested = f"x = f{q}{{'a' if p else {q}b{q}}}{q}"
    multiline = "x = (f\"a {b if c\n     else d}\")"
    # Asserted as "this interpreter would not let it through", not as "the
    # detector flagged it". On 3.11 these do not parse at all, so the detector
    # correctly returns nothing and the real compiler has already done the job;
    # on 3.12+ they parse fine and only the detector stands between them and CI.
    # Asserting the detector directly made this test itself environment-
    # dependent -- it passed locally on 3.13 and failed on CI, which is the
    # exact class of bug this whole file exists to eliminate.
    for src, what in ((nested, "a nested delimiter"), (multiline, "a multi-line expression")):
        assert _rejected_on_311(src), f"{what} would reach CI unnoticed"

    # Valid code must stay silent, including the two shapes most likely to be
    # mistaken for violations: implicit concatenation across lines, and format
    # specs or conversions inside the braces.
    for ok in (f"x = f{q}{{'a' if p else 'b'}}{q}",
               'x = (f"line one {a} "\n     f"line two {b}")',
               'x = f"{a:.2f} {b!r}"'):
        assert not _rejected_on_311(ok), f"flags valid code: {ok!r}"


def test_no_source_file_needs_a_python_newer_than_ci_runs():
    offenders = []
    for path in _project_python_files():
        for lineno, text in _fstrings_needing_312(path.read_text(errors="ignore")):
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
