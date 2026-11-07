#!/usr/bin/env bash
# Rebuild models/ from the Longpipe v0.0.5 weight packs.
#
#   TF_PY=/path/to/tf-env/bin/python MP_PY=/path/to/mp-env/bin/python tools/build_all.sh [tier ...]
#
#   TF_PY : Python with tensorflow (tools/requirements-convert.txt)
#   MP_PY : Python with mediapipe  (tools/requirements-metadata.txt)
#   tiers : any of xs small medium large xl (default: all)
#
# Weights are downloaded to $WEIGHTS_DIR (default: ./weights, git-ignored) and checked
# against the SHA-256 values published in the CDN manifest.
set -euo pipefail
cd "$(dirname "$0")/.."

TF_PY=${TF_PY:-python3}
MP_PY=${MP_PY:-python3}
WEIGHTS_DIR=${WEIGHTS_DIR:-weights}
BASE=https://cdn.longpipe.dev/models/v/0.0.5
[ $# -eq 0 ] && set -- xs small medium large xl

sha_for() {
  case "$1" in
    xs)     echo f1786bf9ffb03916fd265c253f4928ec4a7576855162aacc856aaf90908899e9 ;;
    small)  echo 8fea35e96ee8e37d1b2b6e901a2a7f8df1ce2ec14e1a1227636df6b410e2c11f ;;
    medium) echo 7d8bf5d787da24e679193e70db1e7b9ca62f473ef79d8f14db1604dda8d99dd5 ;;
    large)  echo 40ad8b9f538cf70b89b09a41f916ab441610c9a92ff4222a27071fa22fe92ae9 ;;
    xl)     echo 4f25cb0c267d8966329ea7882afd42a64856786d5de1a276b67eb20f42043fe4 ;;
    *) echo "unknown tier: $1" >&2; exit 1 ;;
  esac
}

mkdir -p "$WEIGHTS_DIR" models
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

for t in "$@"; do
  w="$WEIGHTS_DIR/model_$t.f16.bin"
  [ -f "$w" ] || curl -fsSL -o "$w" "$BASE/model_$t.f16.bin"
  got=$(shasum -a 256 "$w" | cut -d' ' -f1)
  [ "$got" = "$(sha_for "$t")" ] || { echo "checksum mismatch for $w" >&2; exit 1; }

  for o in landscape portrait; do
    flag=""; [ "$o" = portrait ] && flag="--portrait"
    "$TF_PY" tools/convert_to_tflite.py "$w" "$tmp/$o-$t.tflite" --tier "$t" $flag
    "$MP_PY" tools/add_metadata.py "$tmp/$o-$t.tflite" "models/longpipe_${o}_${t}_float16.tflite" > /dev/null
    echo "models/longpipe_${o}_${t}_float16.tflite"
  done
done
