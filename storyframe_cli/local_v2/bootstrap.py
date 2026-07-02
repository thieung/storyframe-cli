from __future__ import annotations

import sys
from pathlib import Path


def add_local_v2_dependency_paths() -> None:
    here = Path(__file__).resolve()
    workspace_root = here.parents[4]
    candidates = [
        workspace_root / "work" / ".deps" / "storyframe-local-v2",
        Path.cwd() / "work" / ".deps" / "storyframe-local-v2",
        workspace_root / "outputs" / "storyframe-cli" / "work" / ".deps" / "storyframe-local-v2",
    ]
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))

