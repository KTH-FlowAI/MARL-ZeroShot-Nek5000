#!/bin/bash -l
#SBATCH -A deepwing
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -N 22
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH -J LC-NP-OM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuninw@umich.edu
#SBATCH --output=log-files/train-%j
#SBATCH --error=log-files/train-%j

source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
export UCX_WARN_UNUSED_ENV_VARS=n
export HWLOC_HIDE_ERRORS=1
#[MOD] Submit from the main path:  sbatch execs/sjob-train-n11.sh
#[MOD] SLURM_SUBMIT_DIR is where you ran `sbatch` (the repo root); fall
#[MOD] back to the script location for a non-SLURM run.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)}"
cd "${ROOT_DIR}"                  #[MOD] run python from the repo root
LOG_DIR="${ROOT_DIR}/log-files"   #[MOD] log-files at the repo root
CACHE_DIR="${ROOT_DIR}/.caches"   #[MOD] RUN_PATH_*.txt cache at the repo root
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes, $SLURM_NPROCS processors."

CONFIG_NAME="conf/MC16-TD3-ng-111.yml"   #[MOD] default is a config PATH relative to the repo root
RUN_MODE="run"
LOAD_AGENT=""   #[MOD] override runner.load_agent (True/False) from CLI; empty = use the config value
# Parse command line arguments
# Usage: ./unified-script --config [CONFIG_NAME] --run-mode [RUN_MODE] --rank [RANK]
# Parse command-line arguments after `--`
while [[ $# -gt 1 ]]; do
    case "$1" in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --load-agent)
            LOAD_AGENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done
#[MOD] Resolve the config to an ABSOLUTE path (accepts a path relative to the
#[MOD] repo root, or an absolute path); fail early if it does not exist.
_cfg_in="${CONFIG_NAME}"
CONFIG_NAME="$(realpath -e "${_cfg_in}" 2>/dev/null)" || {
    echo "[ERR] Config file not found: ${_cfg_in}" >&2; exit 1; }
CONFIG_TAG="$(basename "${CONFIG_NAME}")"   #[MOD] short tag for log filenames
#[MOD] Optional runner.load_agent override (switch policy loading on/off);
#[MOD] empty keeps the config value. Validate to catch typos early.
LOAD_AGENT_ARG=""
if [ -n "${LOAD_AGENT}" ]; then
    if [ "${LOAD_AGENT}" != "True" ] && [ "${LOAD_AGENT}" != "False" ]; then
        echo "[ERR] --load-agent must be True or False, got: ${LOAD_AGENT}" >&2; exit 1
    fi
    LOAD_AGENT_ARG="runner.load_agent=${LOAD_AGENT}"
fi
echo "DRL CONFIG: ${CONFIG_NAME}, RUN MODE: ${RUN_MODE}"

mpirun -n 1 python -m nek_MARL initial ${CONFIG_NAME} \
    > ${LOG_DIR}/log.initial.${CONFIG_TAG} 2>&1

AGENT_RUN_NAME=$(grep -ri 'agent_run_name' ${CONFIG_NAME} | sed -E "s/^[^:]*:[[:space:]]*//; s/[\"']//g; s/[[:space:]]+\$//")
NTOT=$(grep -ri 'nproc' ${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')
RUN_PATH=$(head -n 1 ${CACHE_DIR}/RUN_PATH_${AGENT_RUN_NAME}.txt)
AGENT=$(tail -n 1 ${CACHE_DIR}/RUN_PATH_${AGENT_RUN_NAME}.txt)

echo "RUN_PATH: ${RUN_PATH}, NTOT: ${NTOT}"

#[MOD] History archiving/cleanup is handled inside `initial`
#[MOD] (src/initial.py:preserve_and_clean_train); no shell-side archiving.

mpirun --mca io ompio \
    -n 1 python -m nek_MARL ${RUN_MODE} ${CONFIG_NAME} runner.policy=${AGENT} ${LOAD_AGENT_ARG} :\
    -n ${NTOT} bash -c "cd ${RUN_PATH} && ./nek5000" \
    > ${LOG_DIR}/log.run.${CONFIG_TAG} 2>&1

echo "Finished at $(date)"
