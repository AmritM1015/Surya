#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_CONFIG="easy_inference/config_easy.yaml"

if [[ ! -f "$DEFAULT_CONFIG" ]]; then
  echo "ERROR: missing $DEFAULT_CONFIG"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found in $ROOT_DIR"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python easy_inference/run_easy_inference.py --config-path "$DEFAULT_CONFIG" "$@"
