#!/bin/sh
# Gradi pse-rstdp_<v>_all.deb bez dpkg-deb.
#
# .deb je obicna `ar` arhiva s tri clana u tacno ovom redoslijedu:
#   debian-binary, control.tar.gz, data.tar.gz
# Zato se paket moze sastaviti i tamo gdje dpkg-a nema (npr. u Flatpak
# sandboxu), a rezultat je validan za `sudo dpkg -i` na Mintu/Ubuntu.
set -eu

VERSION="3.0.0"
PKG="pse-rstdp"
ARCH="all"
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD="$HERE/build"
ROOT="$BUILD/root"
OUT="$HERE/${PKG}_${VERSION}_${ARCH}.deb"

rm -rf "$BUILD"
mkdir -p "$ROOT/DEBIAN" \
         "$ROOT/usr/lib/python3/dist-packages/pse_rstdp" \
         "$ROOT/usr/bin" \
         "$ROOT/usr/share/doc/$PKG"

# ---------------- sadrzaj ----------------
cp "$HERE"/pse_rstdp/*.py "$ROOT/usr/lib/python3/dist-packages/pse_rstdp/"
[ -f "$HERE/pse_rstdp/selftest_ref.json" ] && \
  cp "$HERE/pse_rstdp/selftest_ref.json" "$ROOT/usr/lib/python3/dist-packages/pse_rstdp/"

cat > "$ROOT/usr/bin/pse-rstdp" <<'EOF'
#!/usr/bin/python3
import sys
from pse_rstdp.cli import main
sys.exit(main())
EOF
chmod 755 "$ROOT/usr/bin/pse-rstdp"

for f in GUIDE.sr.md GUIDE.en.md README.md; do
  [ -f "$HERE/$f" ] && cp "$HERE/$f" "$ROOT/usr/share/doc/$PKG/"
done
# spec koji kod implementira ide uz paket
for d in "$HERE/.."/PSE_R-STDP_nacrt_v3.docx; do
  [ -f "$d" ] && cp "$d" "$ROOT/usr/share/doc/$PKG/"
done

cat > "$ROOT/usr/share/doc/$PKG/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: $PKG

Files: *
Copyright: $(date +%Y) PSE
License: MIT
EOF

printf '%s (%s) unstable; urgency=low\n\n  * Nacrt v3: zatvoren spec.\n\n -- PSE <pse@localhost>  %s\n' \
  "$PKG" "$VERSION" "$(date -R)" > "$BUILD/changelog.Debian"
gzip -9n -c "$BUILD/changelog.Debian" > "$ROOT/usr/share/doc/$PKG/changelog.Debian.gz"

find "$ROOT" -type d -exec chmod 755 {} +
find "$ROOT" -type f ! -name pse-rstdp -exec chmod 644 {} +
chmod 755 "$ROOT/usr/bin/pse-rstdp"

# ---------------- control ----------------
SIZE=$(du -sk "$ROOT" | cut -f1)
cat > "$ROOT/DEBIAN/control" <<EOF
Package: $PKG
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Depends: python3 (>= 3.8), python3-numpy
Recommends: python3-matplotlib
Installed-Size: $SIZE
Maintainer: PSE <pse@localhost>
Description: R-STDP koincidencijski detektor s Care Invariantom
 Trofaktorsko ucenje (R-STDP) u kojem Care Invariant (w_s, w_o, w_f)
 igra ulogu neuromodulatora.  Mjeri da li tezina koju invariant daje
 buducnosti pomjera domet dodjele kredita unazad (Dr*).
 .
 Cist NumPy, ~110 neurona, stanje modela ispod 1 MB -- radi na masini
 sa 4 GB RAM-a.  Bez PyTorch-a i bez GPU-a.  Deterministicno: isti seed
 daje bit-identican rezultat na svakoj masini, sto je uslov da se
 rezultati s dvije masine smiju spajati u isti dizajn.
 .
 Faze R0-R5 s gate-om na svakom prelazu; falsifikatori se izvrsavaju
 kao provjere prije gledanja rezultata.
EOF

# md5sums
( cd "$ROOT" && find usr -type f -print0 | sort -z | \
  xargs -0 md5sum > DEBIAN/md5sums ) 2>/dev/null || true
chmod 644 "$ROOT/DEBIAN/control" "$ROOT/DEBIAN/md5sums" 2>/dev/null || true

# ---------------- sastavljanje ----------------
cd "$ROOT"
printf '2.0\n' > "$BUILD/debian-binary"
tar --owner=root --group=root --numeric-owner -czf "$BUILD/control.tar.gz" \
    -C "$ROOT/DEBIAN" .
tar --owner=root --group=root --numeric-owner -czf "$BUILD/data.tar.gz" \
    -C "$ROOT" usr

cd "$BUILD"
rm -f "$OUT"
ar rc "$OUT" debian-binary control.tar.gz data.tar.gz

echo "OK -> $OUT"
ls -l "$OUT"
