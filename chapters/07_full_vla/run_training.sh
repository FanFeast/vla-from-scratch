#!/bin/bash
# Long Ch07 training run (~3.2h on an RTX 4090), logged to training_output.log.
# Resolves to this script's own directory, so it works from any checkout.
set -euo pipefail
cd "$(dirname "$0")"

python3 -u train.py --preset so100 "$@" 2>&1 | tee training_output.log
