#!/bin/bash -l
#### A Template for submitting your JOB

#-------- Account ------
#SBATCH -A deepwing

#-------- Partions ------------
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1

#-------- Outputs and notification ------
#SBATCH -J Eval-Enegry
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuninw@umich.edu
#SBATCH --output=./log-files/log-eval-td3-reth856-h169
#SBATCH --error=./log-files/log-eval-td3-reth856-h169
#SBATCH --begin=2025-09-21T10:45:00

# Environment 
#------------------------
#unset MPI_UCX_ROOT UCX_ROOT
export UCX_WARN_UNUSED_ENV_VARS=n
source ~/.bashrc.miniforge
source ~/.bashrc.openmpi_ucx
#[MOD] Submit from the main path:  sbatch execs/sjob-eval.sh
#[MOD] SLURM_SUBMIT_DIR is where you ran `sbatch` (the repo root); fall
#[MOD] back to the script location for a non-SLURM run.
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)}"
cd "${ROOT_DIR}"                  #[MOD] run python from the repo root
LOG_DIR="${ROOT_DIR}/log-files"   #[MOD] log-files at the repo root
CACHE_DIR="${ROOT_DIR}/.caches"   #[MOD] RUN_PATH_*.txt cache at the repo root
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"
#------------------------


# Summary
#------------------------
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
#------------------------

CONFIG_NAME="conf/eval-LCReth310-H172-TD3-1utau.yml"   #[MOD] default is a config PATH relative to the repo root
RUN_MODE="run"
IOSTEP=10000
writeInterval=10000
SMPSTEP=10
reward_alpha=1.0
reward_beta=1.0
reward_gamma=1.0
# Parse command line arguments
# Usage: ./unified-script --config [CONFIG_NAME] --run-mode [RUN_MODE] --rank [RANK]
# Parse command-line arguments after `--`
while [[ $# -gt 1 ]]; do
    case "$1" in
        --config)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --run-mode)
            RUN_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done
#[MOD] Resolve the config to an ABSOLUTE path (accepts a relative or absolute
#[MOD] path); fail early if it does not exist. Downstream uses ${CONFIG_NAME}.
_cfg_in="${CONFIG_NAME}"
CONFIG_NAME="$(realpath -e "${_cfg_in}" 2>/dev/null)" || {
    echo "[ERR] Config file not found: ${_cfg_in}" >&2; exit 1; }
CONFIG_TAG="$(basename "${CONFIG_NAME}")"   #[MOD] short tag for log filenames
echo "DRL CONFIG: ${CONFIG_NAME}, RUN MODE: ${RUN_MODE}"
NTOT=$(grep -ri 'nproc' ${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')

for ienv in {1..6}
do
       echo "Count: $ienv"
       mpirun -n 1 python -m nek_MARL initial $CONFIG_NAME \
        runner.random_init=-1 \
        runner.rank=${ienv} \
        runner.evaluation=True runner.learnt_policy=True \
        runner.load_agent=True \
	simulation.IOSTEP=${IOSTEP} \
	simulation.writeInterval=${writeInterval} \
	simulation.SMPSTEP=${SMPSTEP} 
       
       var=$(grep -ri 'agent_run_name' ${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')
       
       RUN_PATH=$(head -n 1 ${CACHE_DIR}/RUN_PATH_${var}.txt)
       
       # To use mpi_split, one should launch two programs at once. Therefore 
       mpirun --mca pml ucx \
              -n 1 python -m nek_MARL evaluate $CONFIG_NAME \
              runner.rank=${ienv}  runner.random_init=-1\
	      runner.reward_fn=net_gain \
	      runner.reward_alpha=1.0 runner.reward_beta=1.0 runner.reward_gamma=1.0 \
              runner.evaluation=True  \
              runner.load_agent=True :\
              -n ${NTOT} bash -c "cd ${RUN_PATH} && ./nek5000" > ${LOG_DIR}/log.eval.${CONFIG_TAG} 2>&1
done
