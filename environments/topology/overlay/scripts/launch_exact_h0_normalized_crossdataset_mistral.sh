#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/launch_exact_h0_normalized_crossdataset_pilot.sh mistral "${1:-2}"
