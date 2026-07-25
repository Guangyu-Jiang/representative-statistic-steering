#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/launch_exact_h0_crossdataset_model_pilot.sh mistral "${1:-1}"
