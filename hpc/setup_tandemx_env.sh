#!/bin/bash -l
# One-time (or rebuild-on-demand) setup of the TANDEMX Python analysis environment
# on Woody HPC. Submit from the Woody login node:
#   sbatch hpc/setup_tandemx_env.sh
#
# Uses mamba (bootstrapped from conda-forge) rather than plain conda: the classic
# conda solver OOM-killed at ~29GB trying to solve this dependency set in one shot
# (2026-09-04, job 12452756). Mamba solved the identical spec in ~4 min using <4GB.
#SBATCH --job-name=setup_tandemx_env
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/saturn/gwgi/gwgifu0h/00_libs/setup_tandemx_env.log

set -e
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python/3.9-anaconda

CONDA_BASE=/home/saturn/gwgi/gwgifu0h/00_libs/conda_tandemx
mkdir -p "$CONDA_BASE/envs" "$CONDA_BASE/pkgs"
conda config --add envs_dirs "$CONDA_BASE/envs"
conda config --add pkgs_dirs "$CONDA_BASE/pkgs"

echo "=== bootstrapping mamba (fast/low-memory solver) ==="
conda create -y -n mambaboot -c conda-forge mamba
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mambaboot

echo "=== creating TANDEMX env via mamba ==="
mamba create -y -n TANDEMX -c conda-forge \
    python=3.10 \
    rasterio \
    geopandas \
    matplotlib \
    scipy \
    numpy \
    pystac-client \
    s3fs \
    zarr \
    odc-stac \
    xdem \
    geoutils \
    requests \
    shapely \
    pyproj
conda deactivate

echo "=== verifying import ==="
conda activate TANDEMX
python3 -c "
import rasterio, geopandas, xdem, matplotlib, scipy, numpy
import odc.stac, pystac_client, s3fs, zarr
print('rasterio', rasterio.__version__)
print('geopandas', geopandas.__version__)
print('xdem', xdem.__version__)
print('matplotlib', matplotlib.__version__)
print('scipy', scipy.__version__)
print('numpy', numpy.__version__)
print('pystac_client', pystac_client.__version__)
print('s3fs', s3fs.__version__)
print('zarr', zarr.__version__)
print('ALL IMPORTS OK')
"
echo "=== SETUP COMPLETE ==="
