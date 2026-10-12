#!/usr/bin/env bash
# Build a single-file console pse-rstdp.exe (Windows). CLI, so no installer.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; APP="$(dirname "$HERE")"; REPO="$(dirname "$APP")"
PY="${PYTHON:-$APP/.venv/Scripts/python.exe}"
VERSION="${1:-3.0.0}"; PKG="pse-rstdp"
OUT="$REPO/dist"
echo "== PyInstaller onefile"
rm -rf "$APP/build/pyi"
REF_WIN="$(cygpath -w "$APP/pse_rstdp/selftest_ref.json")"
"$PY" -m PyInstaller --noconfirm --clean --onefile --name "PSE-RSTDP-${VERSION}-windows-x64" \
  --icon "$HERE/$PKG.ico" \
  --distpath "$OUT" --workpath "$APP/build/pyi/work" --specpath "$APP/build/pyi" \
  --collect-submodules pse_rstdp --collect-data pse_rstdp \
  --add-data "${REF_WIN};pse_rstdp" \
  "$HERE/entry.py"
echo "built $OUT/PSE-RSTDP-${VERSION}-windows-x64.exe"
