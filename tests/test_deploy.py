"""The deployment guide is prose describing a list of files, which is exactly
the kind of thing that rots: state files get added to .gitignore as the system
grows, and nothing makes anyone revisit a markdown file. A forgotten one is
silent -- the host starts fine and just quietly lacks that piece of history.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_README = ROOT / "deploy" / "README.md"

# Runtime state the host is deliberately NOT given, with the reason. Anything
# gitignored as production state and not listed here must appear in the guide.
EXCLUDED_AND_THE_GUIDE_SAYS_SO = {
    # A reader would reasonably expect to copy this one, so the guide has to
    # say why they should not.
    "scheduler/live_daemon_state.json",
}

EXCLUDED_AND_NOT_WORTH_A_LINE = {
    # Dormant: signal_store.py still defines push_manual_signal and the active-
    # signal helpers, but nothing the daemon runs calls them and neither file
    # exists in production. Named here rather than in the guide so that if this
    # list ever goes stale it goes stale in a test, not in prose.
    "execution/pending_manual_signals.json",
    "execution/active_manual_signals.json",
    # Written by the LOCAL hyperopt runner, on the machine that runs it.
    "execution/freqtrade_userdir/hyperopt_config.json",
}


def _gitignored_production_state() -> list[str]:
    """The paths under .gitignore's 'Production runtime state' heading, plus
    the older runtime-state entries the daemon also rewrites."""
    lines = (ROOT / ".gitignore").read_text().splitlines()
    paths = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.endswith("/"):
            continue
        if line.endswith(".json") and "*" not in line:
            paths.append(line)
    return paths


class TestTheDeploymentGuideStaysTrue:
    def test_every_state_file_is_either_copied_or_explicitly_not(self):
        """A state file that is neither mentioned nor deliberately excluded is
        a file someone will forget to move."""
        guide = DEPLOY_README.read_text()
        runtime_state = [p for p in _gitignored_production_state()
                         if p.startswith(("candidates/", "execution/", "scheduler/", "llm_pipeline/"))
                         or "/" not in p]
        missing = []
        for path in runtime_state:
            if path in EXCLUDED_AND_NOT_WORTH_A_LINE:
                continue
            if path in EXCLUDED_AND_THE_GUIDE_SAYS_SO:
                assert Path(path).name in guide, (
                    f"{path} is excluded on purpose but the guide never says so, and a reader "
                    f"would expect to copy it")
                continue
            # Mentioned either by full path or by bare filename -- the scp
            # commands group by directory, so the filename alone is enough.
            if path not in guide and Path(path).name not in guide:
                missing.append(path)
        assert not missing, (
            f"state files the deployment guide never mentions: {missing}. "
            f"A host missing one starts perfectly and silently lacks that history. "
            f"Add them to step 4, or to INTENTIONALLY_NOT_COPIED with the reason.")

    def test_the_lean_install_omits_exactly_what_the_daemon_never_imports(self):
        """The guide tells the reader to install a specific package list rather
        than requirements.txt, on the claim that freqtrade and scipy are never
        reached by anything the daemon runs. If that stops being true, the
        recommended install silently breaks the host."""
        import ast

        guide = DEPLOY_README.read_text()
        for pkg in ("freqtrade", "scipy"):
            assert pkg in guide, f"the guide no longer explains why {pkg} is omitted"

        # Walk what the daemon imports at MODULE level, transitively, and prove
        # neither package is reached. Source-level, so it needs neither
        # installed. Module level specifically: imports nested inside a function
        # are deliberately lazy throughout this project (freqtrade_bridge and
        # hyperopt_runner both do it) precisely so importing them costs nothing,
        # and flagging those would make this test demand the opposite of the
        # pattern that makes the lean install possible.
        seen: set[str] = set()
        offenders: list[str] = []

        def walk(module: str) -> None:
            if module in seen:
                return
            seen.add(module)
            path = ROOT / (module.replace(".", "/") + ".py")
            if not path.exists():
                return
            for node in ast.parse(path.read_text()).body:  # top level only
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in ("freqtrade", "scipy", "sklearn"):
                        offenders.append(f"{module} imports {name} at module level")
                    if (ROOT / (name.replace(".", "/") + ".py")).exists():
                        walk(name)

        walk("scheduler.live_daemon")
        assert not offenders, (
            f"the daemon now reaches a package the deployment guide tells the host not to install: "
            f"{offenders}. Either update deploy/README.md's install line, or keep the import local.")


def test_preflight_never_writes_state():
    """It runs on a host about to inherit real history. A check that mutates
    what it is checking is worse than no check."""
    import ast

    src = (ROOT / "deploy" / "preflight.py").read_text()
    tree = ast.parse(src)
    banned = {"write_json", "save_horizons", "park_proposal", "unpark_proposal",
              "save_confirmation_prior", "append_trade", "update_trade",
              "record_test_result", "save_battery_state", "mark_escalated"}
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    called |= {node.func.attr for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not (called & banned), f"preflight calls a state-writing function: {called & banned}"
    assert "open(" not in src.replace("shutil.disk_usage", ""), "preflight opens a file for writing"


def test_the_systemd_unit_restarts_and_logs():
    """The whole reason for systemd over `nohup` is that a process dying at 3am
    comes back. A unit without Restart= silently does not."""
    unit = (ROOT / "deploy" / "crypto-agent.service").read_text()
    assert "Restart=always" in unit, "the unit no longer restarts on failure"
    assert "StartLimitBurst" in unit, "a crash-looping daemon would retry forever"
    assert "PYTHONUNBUFFERED=1" in unit, (
        "without this Python buffers print() and the journal looks empty -- "
        "indistinguishable from a hung daemon")
    assert "-m scheduler.live_daemon" in unit
    assert "WantedBy=multi-user.target" in unit, "the service would not start on boot"
