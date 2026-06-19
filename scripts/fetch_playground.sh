#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/angr-binaries"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Playground already exists: $TARGET_DIR"
  exit 0
fi

git clone --depth 1 https://github.com/angr/binaries "$TARGET_DIR"
echo "Playground binaries cloned to $TARGET_DIR"
