#!/bin/bash
# Sync this repo's code/ (and optionally prc07_overview_out/ outputs) to the
# Woody HPC clone, then translate local-machine absolute paths to their HPC
# equivalents. Run this from the local workstation before submitting any HPC
# job that depends on code changes made since the last deploy.
#
# Usage:
#   hpc/deploy_to_hpc.sh            # sync code/ only (fast, do this every time)
#   hpc/deploy_to_hpc.sh --outputs  # also sync prc07_overview_out/ (~750MB+,
#                                   # only needed when local pipeline outputs
#                                   # that HPC scripts depend on have changed)
set -euo pipefail

LOCAL_ROOT="/home/student/Desktop/_0_Korbinian_TANDEM-X/_code"
HPC_ROOT="/home/saturn/gwgi/gwgifu0h/meltrate-klyuchevskaya-glacier-tandemx"

echo "=== syncing code/ ==="
rsync -az "$LOCAL_ROOT/code/" "woody:$HPC_ROOT/code/"

echo "=== translating local paths to HPC paths ==="
ssh woody "bash -lc 'cd $HPC_ROOT && find code -name \"*.py\" | xargs sed -i \
  -e \"s#/media/saturn/01_TDX_data#/home/saturn/gwgi/gwgifu0h/01_TDX_data#g\" \
  -e \"s#$LOCAL_ROOT#$HPC_ROOT#g\" \
  -e \"s#/home/student/anaconda3/envs/TANDEMX#/home/saturn/gwgi/gwgifu0h/00_libs/conda_tandemx/envs/TANDEMX#g\"'"

if [[ "${1:-}" == "--outputs" ]]; then
    echo "=== syncing prc07_overview_out/ (this may take a while) ==="
    rsync -az "$LOCAL_ROOT/prc07_overview_out/" "woody:$HPC_ROOT/prc07_overview_out/"
fi

echo "=== done ==="
