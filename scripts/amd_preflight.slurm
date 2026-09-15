#!/bin/bash
#SBATCH --job-name=fly-plastic-preflight
#SBATCH --partition=gpuworker
#SBATCH --account=posttraining
#SBATCH --wckey=v1/model_development/sft/posttraining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --exclude=tus1-p16-g41,tus1-p13-g23
set -euo pipefail
PROJECT_DIR="${HOME}/fly_wordbrain"
IMAGE_PATH=/home/teams/posttraining/sqsh/amdagi_verl-dev_rocm7.14_torch2.12_release_0724.sqsh
mkdir -p "${PROJECT_DIR}/results/amd-preflight"
sha256sum "${IMAGE_PATH}" > "${PROJECT_DIR}/results/amd-preflight/image-${SLURM_JOB_ID}.sha256"
srun --nodes=1 --ntasks=1 --gpus-per-node=1 --cpu-bind=cores \
  --container-image="${IMAGE_PATH}" \
  --container-mounts="${PROJECT_DIR}:/workspace" \
  --container-workdir=/workspace \
  /opt/venv/bin/python -u scripts/amd_preflight.py \
  --output "results/amd-preflight/runtime-${SLURM_JOB_ID}.json"
