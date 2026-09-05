#!/bin/bash
# Submit a single analysis script as a SLURM job on Woody HPC, from the local
# workstation. Handles both local-file-only scripts and scripts that fetch
# external data (Sentinel-2/Landsat/ITS_LIVE/ERA5) at runtime -- the proxy
# is exported either way, since it's a no-op for scripts that don't need it.
#
# Usage:
#   hpc/run_on_hpc.sh ash_analysis/28_hotspot_coldspot_analysis.py
#   hpc/run_on_hpc.sh dH_robust_all.py --cpus 8 --time 02:00:00
#
# Run hpc/deploy_to_hpc.sh first if the script (or anything it imports) has
# changed locally since the last deploy -- this does not sync code itself.
set -euo pipefail

SCRIPT_REL="$1"; shift || true
CPUS=4
TIME=01:00:00
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpus) CPUS="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

HPC_ROOT="/home/saturn/gwgi/gwgifu0h/meltrate-klyuchevskaya-glacier-tandemx"
JOB_NAME=$(basename "$SCRIPT_REL" .py)
SCRIPT_DIR=$(dirname "$SCRIPT_REL")
LOG="/home/saturn/gwgi/gwgifu0h/00_libs/run_${JOB_NAME}.log"

SBATCH_SCRIPT=$(mktemp)
cat > "$SBATCH_SCRIPT" <<EOF
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG}

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
module load python/3.9-anaconda
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate TANDEMX
cd "${HPC_ROOT}/code/${SCRIPT_DIR}"
python3 -u "$(basename "$SCRIPT_REL")"
EOF

scp -q "$SBATCH_SCRIPT" "woody:/home/saturn/gwgi/gwgifu0h/00_libs/run_${JOB_NAME}.sh"
rm -f "$SBATCH_SCRIPT"
echo "=== submitting ${SCRIPT_REL} (cpus=${CPUS} time=${TIME}) ==="
ssh woody "sbatch /home/saturn/gwgi/gwgifu0h/00_libs/run_${JOB_NAME}.sh"
echo "log: ${LOG} (on woody)"
