"""Crash-safe JSON persistence, shared by every state file in this
project (production and replay alike).

Why this exists. Every state file was written with a bare
`path.write_text(json.dumps(...))`. That is not one operation: the file
is truncated to zero, then the new bytes are written. A crash, a kill, a
full disk or a power loss in between leaves a TRUNCATED file, and the
next `json.loads` raises -- taking down whichever component happens to
read it first, with the original contents already gone.

The exposure is real rather than theoretical. `candidates/status_history.json`
is ~160KB and is rewritten on every single candidate status change (six
static plus every dynamic candidate, every weekly run). `replay/state/`'s
checkpoint is rewritten after EVERY simulated day, specifically so a
crash can resume -- a guarantee that a corrupt checkpoint would invert
into "resume is now impossible". Losing either means losing the record
of what this project has actually tested, which is the one thing it
cannot reconstruct from anywhere else.

`write_json` writes to a temporary file in the SAME directory and then
`os.replace`s it over the target. `os.replace` is atomic on POSIX and on
Windows: a reader sees either the complete old file or the complete new
one, never a half-written one. Same directory matters -- `os.replace`
across filesystems is not atomic.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any, *, indent: int | None = 2, default=None) -> None:
    """Atomically replace `path` with `data` as JSON.

    `default` is passed through to `json.dumps` (several callers persist
    pandas Timestamps and rely on `default=str`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent, default=default)
            f.flush()
            # fsync before the rename: without it the rename can land while the
            # new file's own contents are still only in the page cache, which
            # turns "atomic replace" into "atomically replaced with nothing"
            # after a power loss.
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave a stray .tmp behind on failure -- the target file is
        # untouched either way, which is the whole point.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any) -> Any:
    """Read JSON, returning `default` if the file doesn't exist.

    A file that exists but is unreadable is NOT silently treated as
    missing: that would turn "state got corrupted" into "there is no
    state", quietly discarding real history and starting over. Corruption
    should be loud.
    """
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())
