#!/bin/bash -l
#SBATCH -A deepwing
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=41
#SBATCH --cpus-per-task=1
#SBATCH -J ng-111
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuninw@umich.edu
#SBATCH --output=log-files/ng-111-%j.out
#SBATCH --error=log-files/ng-111-%j.err

source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
export UCX_WARN_UNUSED_ENV_VARS=n
export HWLOC_HIDE_ERRORS=1
LOG_DIR="log-files"
mkdir -p ${LOG_DIR}

echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes, $SLURM_NPROCS processors."

CONFIG_NAME="MC16-TD3-ng-111.yml"
RUN_MODE="run"
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
echo "DRL CONFIG: ${CONFIG_NAME}, RUN MODE: ${RUN_MODE}"

mpirun -n 1 python -m nek_MARL initial ../conf/${CONFIG_NAME} \
    > ${LOG_DIR}/log.initial.${CONFIG_NAME} 2>&1

# agent_run_name is now a string: take everything after the first ':', strip spaces and quotes
AGENT_RUN_NAME=$(grep -ri 'agent_run_name' ../conf/${CONFIG_NAME} | sed -E "s/^[^:]*:[[:space:]]*//; s/[\"']//g; s/[[:space:]]+\$//")
NTOT=$(grep -ri 'nproc' ../conf/${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')
RUN_PATH=$(head -n 1 RUN_PATH_${AGENT_RUN_NAME}.txt)
AGENT=$(tail -n 1 RUN_PATH_${AGENT_RUN_NAME}.txt)

echo "RUN_PATH: ${RUN_PATH}, NTOT: ${NTOT}"

fileNUM=$(ls "${RUN_PATH}"/round* -dq 2>/dev/null | wc -l)
fileNUM=$((fileNUM + 1))
printf -v fileNUM "%03d" "$fileNUM"
mv "${RUN_PATH}/history" "${RUN_PATH}/round${fileNUM}" 2>/dev/null || true
echo "History archived to round${fileNUM}"

mpirun --mca io ompio \
    -n 1 python -m nek_MARL ${RUN_MODE} ../conf/${CONFIG_NAME} runner.policy=${AGENT} :\
    -n ${NTOT} bash -c "cd ${RUN_PATH} && ./nek5000" \
    > ${LOG_DIR}/log.run.${CONFIG_NAME} 2>&1

echo "Finished at $(date)"
