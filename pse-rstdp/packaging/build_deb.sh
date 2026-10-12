#!/usr/bin/env bash
# Build pse-rstdp_<version>_amd64.deb (self-contained PyInstaller bundle, CLI).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; APP="$(dirname "$HERE")"; REPO="$(dirname "$APP")"
PY="${PYTHON:-$APP/.venv/bin/python}"
VERSION="${1:-3.0.0}"; ARCH="${ARCH:-amd64}"; PKG="pse-rstdp"
STAGE="$HERE/stage"; OUT="$REPO/dist"
echo "== PyInstaller bundle"
rm -rf "$APP/build/pyi" "$STAGE"
"$PY" -m PyInstaller --noconfirm --clean --onedir --name "$PKG" \
  --distpath "$APP/build/pyi/dist" --workpath "$APP/build/pyi/work" --specpath "$APP/build/pyi" \
  --collect-submodules pse_rstdp --collect-data pse_rstdp \
  --add-data "$APP/pse_rstdp/selftest_ref.json:pse_rstdp" \
  "$HERE/entry.py"
APPDIR="$STAGE/opt/$PKG"; mkdir -p "$APPDIR"
cp -a "$APP/build/pyi/dist/$PKG/." "$APPDIR/"
cp "$REPO/LICENSE" "$APPDIR/LICENSE"; cp "$REPO/NOTICE" "$APPDIR/NOTICE"
mkdir -p "$STAGE/usr/bin" "$STAGE/usr/share/doc/$PKG" "$STAGE/DEBIAN"
ln -s "/opt/$PKG/$PKG" "$STAGE/usr/bin/$PKG"
cp "$REPO/LICENSE" "$STAGE/usr/share/doc/$PKG/LICENSE"; cp "$REPO/NOTICE" "$STAGE/usr/share/doc/$PKG/NOTICE"
SIZE="$(du -sk "$STAGE/opt" "$STAGE/usr" | awk '{s+=$1} END{print s}')"
cat > "$STAGE/DEBIAN/control" <<CTL
Package: $PKG
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Installed-Size: $SIZE
Maintainer: Andrej Ristic <andrej.s.ristic@gmail.com>
Depends: libc6 (>= 2.34)
Homepage: https://github.com/andrejristic/pse-r
Description: PSE R-STDP - reward-modulated STDP on a small spiking network
 A command-line research tool for reward-modulated spike-timing-dependent
 plasticity experiments on a small spiking neural network. Run phases, seeds
 and a self-test from the terminal: pse-rstdp --help.
CTL
echo "== dpkg-deb"; mkdir -p "$OUT"
DEB="$OUT/${PKG}_${VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group -Zxz --build "$STAGE" "$DEB"; echo "built $DEB"
