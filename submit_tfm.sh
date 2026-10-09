#!/bin/bash
# ---------------------------------------------------------------------------
# Trabajo BATCH: construye y despliega 'tfm:ngc-arm64' en hennessy.
# Usa el HOME de hennessy (ya existe y esta compartido por NFS), asi que
# el Dockerfile y el log son visibles desde el login en ~/hennessy/tfm/.
#
# Enviar:  sbatch ~/hennessy/tfm/submit_tfm.sh
#   (o desde el login:  sbatch /machines/hennessy/home/mariag/tfm/submit_tfm.sh)
# Cola:    squeue -p hennessy-benchmark,hennessy-test
# Log:     tail -f ~/hennessy/tfm/slurm-<JOBID>.out
# ---------------------------------------------------------------------------

#SBATCH -p hennessy-benchmark
#SBATCH --chdir=/home/mariag/tfm
#SBATCH --cpus-per-task=14
#SBATCH -J tfm-build
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maria.garciaminarro@digitaldatafarm.com

set -euo pipefail

IMAGE="tfm:ngc-arm64"
CONTAINER="tfm"

echo "== Trabajo ${SLURM_JOB_ID} en nodo: $(hostname) =="
echo "== Fecha: $(date) =="
echo "== Directorio: $(pwd) =="

# 0. Comprobar Docker
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: no puedo usar Docker en $(hostname). Revisa permisos (grupo docker)."
    exit 1
fi

# 1. Construir la imagen (contexto = ~/tfm, con el Dockerfile)
echo "== Construyendo imagen ${IMAGE} =="
docker build -t "${IMAGE}" .

# 2. Desplegar el contenedor (persiste tras el job; monta tu home tfm)
echo "== Desplegando contenedor ${CONTAINER} =="
docker rm -f "${CONTAINER}" 2>/dev/null || true
docker run -d \
    --name "${CONTAINER}" \
    --gpus all \
    -v /home/mariag/tfm:/workspace/tfm \
    "${IMAGE}" \
    tail -f /dev/null

echo "== Estado del contenedor =="
docker ps --filter "name=${CONTAINER}"

# 3. Validacion critica: compila y ejecuta kernels PyTorch/Triton/TLX/Helion en GPU
echo "== Validacion GPU (validar_gpu.py) =="
docker exec "${CONTAINER}" python /workspace/tfm/validar_gpu.py || echo "AVISO: la validacion GPU ha fallado (ver arriba)"

echo "== Finalizado: $(date) =="
