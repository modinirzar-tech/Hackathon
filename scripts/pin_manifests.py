#!/usr/bin/env python3
"""Re-pin signed MCP manifests after an intentional server/tool change (review the diff first!)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from assistant.config import Settings
s = Settings()
s.manifest_lock_path.unlink(missing_ok=True)
from assistant.runtime import build_runtime
build_runtime(s)
print("pinned ->", s.manifest_lock_path)
