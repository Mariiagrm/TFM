#!/bin/bash
# ---------------------------------------------------------------------------
# Trabajo BATCH: obtiene el BASELINE de rendimiento (matmul Triton con
# @triton.autotune vs cuBLAS) en la GPU de hennessy.
#
#   1. Comprueba Docker, la imagen tfm:ngc-arm64 y que la GPU esta libre.
#   2. Ejecuta validar_gpu.py (PyTorch/Triton/TLX/Helion en GPU).
#   3. Ejecuta baseline_matmul.py para cada dtype.
#
# Enviar:  sbatch ~/hennessy/tfm/submit_baseline.sh
#   (o desde el login:  sbatch /machines/hennessy/home/mariag/tfm/submit_baseline.sh)
# Cola:    squeue -p hennessy-benchmark,hennessy-test
# Log:     tail -f ~/hennessy/tfm/slurm-baseline-<JOBID>.out
# Result.: ~/hennessy/tfm/results/baseline_matmul_<dtype>.{csv,json}
#
# Opcional (variables de entorno al enviar):
#   SIZES="4096 8192 16384" DTYPES="fp16" sbatch submit_baseline.sh
# ---------------------------------------------------------------------------

#SBATCH -p hennessy-benchmark
#SBATCH --chdir=/home/mariag/tfm
#SBATCH --cpus-per-task=14
#SBATCH -J tfm-baseline
#SBATCH -o slurm-baseline-%j.out
#SBATCH -e slurm-baseline-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maria.garciaminarro@digitaldatafarm.com

set -euo pipefail

IMAGE="tfm:ngc-arm64"
SIZES="${SIZES:-4096 8192}"
DTYPES="${DTYPES:-fp16 bf16}"

echo "== Trabajo ${SLURM_JOB_ID} en nodo: $(hostname) =="
echo "== Fecha: $(date) =="
echo "== Directorio: $(pwd) | tamanos: ${SIZES} | dtypes: ${DTYPES} =="

# 0. Comprobaciones previas
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: no puedo usar Docker en $(hostname). Revisa permisos (grupo docker)."
    exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "ERROR: no existe la imagen ${IMAGE}. Construyela antes con: sbatch submit_tfm.sh"
    exit 1
fi

# La GPU debe estar libre para que las medidas sean fiables.
echo "== Estado de la GPU =="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,temperature.gpu --format=csv || true
OCUPADA=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
if [ -n "${OCUPADA}" ]; then
    echo "AVISO: hay procesos usando la GPU; las medidas pueden no ser fiables:"
    echo "${OCUPADA}"
fi

# Ejecuta un comando python en un contenedor efimero con ~/tfm montado.
# --user: resultados a nombre de mariag (no root). HOME=/tmp: cache de Triton escribible.
# USER/LOGNAME: el UID no existe en el /etc/passwd de la imagen y getpass.getuser()
# (lo usa TorchInductor, que usa Helion) fallaria con "getpwuid(): uid not found".
run_py() {
    docker run --rm --gpus all \
        -v /home/mariag/tfm:/workspace/tfm \
        --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
        "${IMAGE}" python "$@"
}

# 1. Validacion (si falla, no tiene sentido medir)
echo "== Validacion GPU (validar_gpu.py) =="
if ! run_py /workspace/tfm/validar_gpu.py; then
    echo "ERROR: la validacion GPU ha fallado; se aborta el baseline."
    exit 1
fi

# 2. Baseline para cada dtype
for DT in ${DTYPES}; do
    echo "== Baseline matmul ${DT} (tamanos: ${SIZES}) =="
    # shellcheck disable=SC2086  # SIZES se expande a varios argumentos a proposito
    run_py /workspace/tfm/baseline_matmul.py --dtype "${DT}" --sizes ${SIZES}
done

echo "== Resultados =="
ls -l results/
echo "== Finalizado: $(date) =="
