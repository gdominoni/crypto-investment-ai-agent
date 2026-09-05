#!/usr/bin/env bash
# Compile every source file with the REAL Python 3.11, the version CI runs.
#
#     ./scripts/check_py311.sh
#
# Why this exists. This project promises "Python 3.11+" and CI enforces it, but
# development happens on 3.13, which accepts syntax 3.11 rejects -- PEP 701
# f-strings above all. That gap has now produced two red-CI incidents, and the
# second one slipped past a hand-written guard that only knew about the first
# one's symptom. A guard that enumerates symptoms will keep missing new ones;
# the compiler cannot.
#
# tests/test_python_compatibility.py still runs in the normal suite as a fast
# pre-filter with no Docker dependency. This is the authority when they disagree.
set -euo pipefail

IMAGE="python:3.11-slim"   # same minor version CI pins
cd "$(dirname "$0")/.."

if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running -- start it, or rely on CI for this check." >&2
    exit 2
fi

docker run --rm -v "$PWD":/src -w /src "$IMAGE" python -c '
import pathlib, sys

# Directories that are not this project`s code. Third-party sources routinely
# need a newer Python than we do and are not ours to fix.
SKIP = {".git", ".venv", "venv", "__pycache__", "freqtrade_userdir", "node_modules"}
errors = []
checked = 0
for path in sorted(pathlib.Path(".").rglob("*.py")):
    if SKIP & set(path.parts):
        continue
    checked += 1
    try:
        compile(path.read_text(errors="ignore"), str(path), "exec")
    except SyntaxError as e:
        errors.append(f"  {path}:{e.lineno}: {e.msg}")

print(f"Python {sys.version.split()[0]}: compiled {checked} files, {len(errors)} error(s)")
for line in errors:
    print(line)
sys.exit(1 if errors else 0)
'
