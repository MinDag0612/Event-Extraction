#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root (or any directory) with:
# bash data/raw/BKEE/download.sh

REPOSITORY_URL="https://github.com/nhungnt7/BKEE.git"
TARGET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bkee.XXXXXX")"

cleanup() {
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

command -v git >/dev/null || {
    echo "git is required to download BKEE." >&2
    exit 1
}

echo "Downloading BKEE from the official repository..."
git clone --depth 1 "$REPOSITORY_URL" "$TEMP_DIR/source"

echo "Copying processed train/dev/test data to $TARGET_DIR..."
cp "$TEMP_DIR/source/processed/train.json" "$TARGET_DIR/train.json"
cp "$TEMP_DIR/source/processed/dev.json" "$TARGET_DIR/dev.json"
cp "$TEMP_DIR/source/processed/test.json" "$TARGET_DIR/test.json"
cp "$TEMP_DIR/source/LICENSE" "$TARGET_DIR/LICENSE"

echo "BKEE download complete."
