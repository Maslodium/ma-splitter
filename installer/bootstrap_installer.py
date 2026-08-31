r"""
M-A Splitter Windows bootstrap installer.

This file is packaged into a small EXE. It copies the app payload into
%LOCALAPPDATA%\M-A Splitter, creates a venv, scans the system, and downloads the
matching ML dependencies.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "M-A Splitter"
INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_NAME


def payload_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "payload"
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def has_nvidia_gpu() -> bool:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def find_host_python() -> list[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable]

    for cmd in (["py", "-3.12"], ["py", "-3"], ["python"], ["python3"]):
        try:
            proc = subprocess.run(
                [*cmd, "-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
            if proc.returncode == 0:
                return cmd
        except Exception:
            pass
    raise RuntimeError(
        "Python 3.10+ was not found. Install Python from https://www.python.org/downloads/windows/ "
        "and enable 'Add python.exe to PATH', then run this installer again."
    )


def copy_payload(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ignored = {
        ".git",
        ".venv",
        "venv",
        "build",
        "dist",
        "payload",
        "__pycache__",
        "input",
        "output",
        "separated",
        "models",
        "checkpoints",
        "gui_settings.json",
        "Install M-A Splitter.exe",
    }
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


def create_shortcut(target_pyw: Path, script: Path) -> None:
    if os.name != "nt":
        return
    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    shortcut_path = desktop / f"{APP_NAME}.lnk"
    ps = f"""
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut('{shortcut_path}')
$shortcut.TargetPath = '{target_pyw}'
$shortcut.Arguments = '"{script}"'
$shortcut.WorkingDirectory = '{script.parent}'
$shortcut.Description = 'Launch {APP_NAME}'
$shortcut.IconLocation = '{target_pyw},0'
$shortcut.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps], check=False)


def install_dependencies(app_dir: Path) -> None:
    venv = app_dir / ".venv"
    py = venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"
    pyw = venv / "Scripts" / "pythonw.exe" if os.name == "nt" else py

    if not py.exists():
        run([*find_host_python(), "-m", "venv", str(venv)])

    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    if has_nvidia_gpu():
        print("[scan] NVIDIA GPU detected; installing CUDA 12.6 Torch wheels.")
        run([
            str(py), "-m", "pip", "install", "--force-reinstall",
            "torch==2.6.0", "torchaudio==2.6.0",
            "--index-url", "https://download.pytorch.org/whl/cu126",
        ])
    else:
        print("[scan] NVIDIA GPU not detected; installing CPU Torch wheels.")

    run([str(py), "install.py"], cwd=app_dir)
    create_shortcut(pyw, app_dir / "gui.py")


def main() -> int:
    print(f"{APP_NAME} installer")
    src = payload_root()
    print(f"[copy] {src} -> {INSTALL_DIR}")
    copy_payload(src, INSTALL_DIR)
    install_dependencies(INSTALL_DIR)
    print(f"\nInstalled to: {INSTALL_DIR}")
    print(f"Desktop shortcut: {APP_NAME}.lnk")
    input("\nPress Enter to exit...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
