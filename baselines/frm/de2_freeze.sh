#!/bin/bash
# Data efficiency of FRM: copy the code of the runs (frm/fit.py, de2_adapter.py, de2_setup.py, de2.sbatch and clock.py)
# to runs/frm/de2-v99/code/<UTC stamp>/, read-only, and print that folder. frm/de2.sbatch jobs run this copy, so later
# edits cannot change a queued job. Usage: CODE=$(bash baselines/frm/de2_freeze.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
CODE=$PWD/runs/frm/de2-v99/code/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $CODE/frm
cp clock.py $CODE/
cp frm/fit.py frm/de2_adapter.py frm/de2_setup.py frm/de2.sbatch $CODE/frm/
(cd .. && git rev-parse HEAD) > $CODE/commit.txt
(cd $CODE && sha256sum clock.py frm/*) > $CODE/sha256.txt
chmod -R a-w $CODE
echo $CODE
