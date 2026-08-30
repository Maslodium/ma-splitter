#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  python3 -m venv "${ROOT}/.venv"
fi

"${PYTHON}" -m pip install --upgrade pip setuptools wheel pyinstaller

PAYLOAD="${ROOT}/payload"
rm -rf "${PAYLOAD}"
mkdir -p "${PAYLOAD}"

rsync -a "${ROOT}/" "${PAYLOAD}/" \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude "build" \
  --exclude "dist" \
  --exclude "payload" \
  --exclude "__pycache__" \
  --exclude "*.spec" \
  --exclude "input" \
  --exclude "output" \
  --exclude "separated" \
  --exclude "models" \
  --exclude "checkpoints" \
  --exclude "gui_settings.json"

"${PYTHON}" -m PyInstaller \
  --name "Install M-A Splitter macOS" \
  --onefile \
  --windowed \
  --add-data "${PAYLOAD}:payload" \
  "${ROOT}/installer/bootstrap_installer_macos.py"

cd "${ROOT}/dist"
ditto -c -k --keepParent "Install M-A Splitter macOS.app" "M-A-Splitter-macOS.zip"
echo "Built: ${ROOT}/dist/M-A-Splitter-macOS.zip"
