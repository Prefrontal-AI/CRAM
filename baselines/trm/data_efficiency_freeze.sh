#!/bin/bash
# Copy the TRM fit, score and unseen-score code to experiments/runs/trm/smalldata/src/<UTC stamp>/ and print that folder.
# The jobs of trm/data_efficiency.sbatch run this copy, so later edits cannot change a running job.
# TRM itself (the clone in third_party/trm, TinyRecursiveModels c011037) is not copied.
# Usage: SRC=$(bash trm/data_efficiency_freeze.sh)
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
SRC=$REPO/experiments/runs/trm/smalldata/src/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$SRC/baselines/trm/trm_shim"
cp "$REPO/baselines/clock.py" "$SRC/baselines/"
cp fit.py unseen.py trm_bench.py trm_configs.sh trm_smalldata.py "$SRC/baselines/trm/"
cp trm_shim/*.py "$SRC/baselines/trm/trm_shim/"
(cd "$REPO" && git rev-parse HEAD) > "$SRC/commit.txt"
(cd "$SRC" && find . -name '*.py' -o -name '*.sh' | sort | xargs sha256sum) > "$SRC/sha256.txt"
echo "$SRC"
