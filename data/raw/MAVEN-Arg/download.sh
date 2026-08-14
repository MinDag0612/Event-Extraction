#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root (or any directory) with:
# uv run bash data/raw/MAVEN-Arg/download.sh

DATA_URL="https://cloud.tsinghua.edu.cn/f/8b3e416fb264400c8a9b/?dl=1"
TARGET_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="$(mktemp "${TMPDIR:-/tmp}/maven-arg.XXXXXX.zip")"

cleanup() {
    rm -f -- "$ARCHIVE"
}
trap cleanup EXIT

command -v curl >/dev/null || {
    echo "curl is required to download MAVEN-Arg." >&2
    exit 1
}
command -v unzip >/dev/null || {
    echo "unzip is required to extract MAVEN-Arg." >&2
    exit 1
}

echo "Downloading MAVEN-Arg..."
curl --fail --location --retry 3 --output "$ARCHIVE" "$DATA_URL"

echo "Extracting data to $TARGET_DIR..."
unzip -j -o "$ARCHIVE" -d "$TARGET_DIR"
rm -f "$TARGET_DIR"/._*.jsonl
echo "MAVEN-Arg download complete."


