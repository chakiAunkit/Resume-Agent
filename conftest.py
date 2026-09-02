"""Repo-root conftest: put the project root on sys.path so tests can
`import src.models` etc. regardless of how pytest is invoked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
