#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="${ROOT}/dist"
PKG="${DIST}/M-A-Splitter-linux"

rm -rf "${PKG}"
mkdir -p "${PKG}"

rsync -a "${ROOT}/" "${PKG}/" \
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

chmod +x "${PKG}/installer/bootstrap_installer_linux.py"
cat > "${PKG}/install.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
python3 installer/bootstrap_installer_linux.py
EOF
chmod +x "${PKG}/install.sh"

cd "${DIST}"
tar -czf "M-A-Splitter-linux.tar.gz" "M-A-Splitter-linux"
echo "Built: ${DIST}/M-A-Splitter-linux.tar.gz"
