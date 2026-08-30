"""
Install runtime dependencies for M-A Splitter.

Run inside an activated virtual environment:

    python install.py

The project intentionally installs basic-pitch with --no-deps because its
package metadata still pulls an older TensorFlow/numpy stack, while this app
uses the ONNX backend with explicitly pinned runtime dependencies.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LOCK = HERE / "requirements-lock.txt"


def run(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.check_call(args)


def require_import(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"[error] import check failed: {name}")
    print(f"[ok] {name}")


def main() -> int:
    py = sys.executable
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([py, "-m", "pip", "install", "-r", str(LOCK)])
    run([py, "-m", "pip", "install", "basic-pitch==0.4.0", "--no-deps"])

    for module in ("torch", "torchaudio", "demucs", "librosa", "pretty_midi", "basic_pitch"):
        require_import(module)
    print("\nM-A Splitter dependencies are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
