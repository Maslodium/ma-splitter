"""
M-A Splitter macOS bootstrap installer.

Packaged as a small .app on macOS. It copies the app into
~/Applications/M-A Splitter, creates a venv and installs dependencies.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path


APP_NAME = "M-A Splitter"
INSTALL_DIR = Path.home() / "Applications" / APP_NAME


def payload_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "payload"
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def find_host_python() -> list[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable]
    for cmd in (["python3.12"], ["python3.11"], ["python3"], ["python"]):
        try:
            proc = subprocess.run(
                [*cmd, "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode == 0:
                return cmd
        except Exception:
            pass
    raise RuntimeError("Python 3.10+ was not found.")


def copy_payload(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ignored = {".git", ".venv", "venv", "build", "dist", "payload", "__pycache__"}
    for item in src.iterdir():
        if item.name in ignored or item.name.endswith(".spec"):
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def create_launcher(app_dir: Path, py: Path) -> None:
    launcher = Path.home() / "Desktop" / f"{APP_NAME}.command"
    launcher.write_text(
        "#!/bin/zsh\n"
        f"cd {sh_quote(str(app_dir))}\n"
        f"exec {sh_quote(str(py))} {sh_quote(str(app_dir / 'gui.py'))}\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_dependencies(app_dir: Path) -> None:
    venv = app_dir / ".venv"
    py = venv / "bin" / "python"
    if not py.exists():
        run([*find_host_python(), "-m", "venv", str(venv)])
    run([str(py), "install.py"], cwd=app_dir)
    create_launcher(app_dir, py)


def main() -> int:
    print(f"{APP_NAME} macOS installer")
    src = payload_root()
    print(f"[copy] {src} -> {INSTALL_DIR}")
    copy_payload(src, INSTALL_DIR)
    install_dependencies(INSTALL_DIR)
    print(f"\nInstalled to: {INSTALL_DIR}")
    print(f"Desktop launcher: {APP_NAME}.command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
