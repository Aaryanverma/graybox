"""Regression: importing/starting the CLI must not load litellm.

litellm and its ~900 modules are lazy-imported, and only loaded when an
LLM-using command actually runs (organize/ask/chat/rebuild-index/refresh).
The checks run in a fresh subprocess so litellm already imported by other
tests can't mask a regression.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

_CHECK_IMPORT = (
    "import sys\n"
    "import graybox.cli\n"
    "loaded = [m for m in sys.modules if m == 'litellm' or m.startswith('litellm.')]\n"
    "assert not loaded, f'importing graybox.cli loaded litellm: {loaded}'\n"
)

_CHECK_START = (
    "import sys\n"
    "import graybox.cli\n"
    "try:\n"
    "    graybox.cli.main(['--help'])\n"
    "except SystemExit:\n"
    "    pass\n"
    "loaded = [m for m in sys.modules if m == 'litellm' or m.startswith('litellm.')]\n"
    "assert not loaded, f'starting the CLI loaded litellm: {loaded}'\n"
)


def _run_in_subprocess(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


class TestCliStartupDoesNotLoadLiteLLM:
    def test_importing_cli_does_not_load_litellm(self):
        _run_in_subprocess(_CHECK_IMPORT)

    def test_starting_cli_does_not_load_litellm(self):
        _run_in_subprocess(_CHECK_START)
