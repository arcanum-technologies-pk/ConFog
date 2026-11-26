#!/usr/bin/env bash
# Downloads CERT Insider Threat r4.2, r5.2, r6.2 + answer key from CMU KiltHub (figshare article 12841247),
# verifies SHA-256 checksums and extracts into data/raw/cert/. Safe to re-run: downloads resume (wget -c)
# and finished extractions are skipped.
#   bash scripts/download_cert.sh            # keeps the .tar.bz2 archives
#   DELETE_ARCHIVES=1 bash scripts/download_cert.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DST=data/raw/cert
ARC=data/raw/archives
mkdir -p "$DST" "$ARC"
declare -A ID=( [answers.tar.bz2]=24857828 [r4.2.tar.bz2]=24856766 [r5.2.tar.bz2]=24849938
                [r6.2.tar.bz2]=24844280 [CHECKSUMS-sha256.txt]=24857831 )
for f in CHECKSUMS-sha256.txt answers.tar.bz2 r4.2.tar.bz2 r5.2.tar.bz2 r6.2.tar.bz2; do
  name=${f%.tar.bz2}
  if [[ -f "$DST/.extracted_$name" ]]; then echo "[cert] $name already extracted - skip"; continue; fi
  echo "[cert] downloading $f"
  wget -c --tries=20 --timeout=60 -O "$ARC/$f" "https://ndownloader.figshare.com/files/${ID[$f]}"
  [[ $f == CHECKSUMS-sha256.txt ]] && continue
  echo "[cert] verifying $f"
  sum=$(grep -E "[ *]$f\$" "$ARC/CHECKSUMS-sha256.txt" | awk '{print $1}')
  if [[ -n "$sum" ]]; then
    echo "$sum  $ARC/$f" | sha256sum -c - || { echo "checksum mismatch for $f - delete it and re-run"; exit 1; }
  else
    echo "[cert] WARNING: no checksum listed for $f"
  fi
  echo "[cert] extracting $f (this can take a long time for r6.2)"
  DEC=$(command -v lbzip2 || command -v pbzip2 || echo bzip2)
  tar -I "$DEC" -xf "$ARC/$f" -C "$DST"
  # normalise layout to data/raw/cert/<release>/{logon,device,file,email,http}.csv + LDAP/
  if [[ $name != answers && ! -f "$DST/$name/logon.csv" ]]; then
    found=$(dirname "$(find "$DST" -path "*$name*" -name logon.csv | head -1)")
    [[ -n "$found" && "$found" != "$DST/$name" ]] && mv "$found" "$DST/$name"
  fi
  touch "$DST/.extracted_$name"
  [[ "${DELETE_ARCHIVES:-0}" == 1 ]] && rm -f "$ARC/$f"
done
ls -la "$DST"
echo "[cert] done"
