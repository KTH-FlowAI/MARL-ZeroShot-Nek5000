#!/bin/bash -l
#SBATCH -A deepwing
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=11
#SBATCH --cpus-per-task=1
#SBATCH -J ng-011
#SBATCH --mail-type=ALL
#SBATCH --mail-user=polsm@kth.se
#SBATCH --output=log-files/ng-011-%j.out
#SBATCH --error=log-files/ng-011-%j.err

source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
export UCX_WARN_UNUSED_ENV_VARS=n
export HWLOC_HIDE_ERRORS=1
export UCX_TLS=sm,self,tcp,cma,sysv,posix
export OMPI_MCA_btl=self,vader,tcp
LOG_DIR="log-files"
mkdir -p ${LOG_DIR}

echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes, $SLURM_NPROCS processors."

CONFIG_NAME="MC16-TD3-ng-011.yml"
RUN_MODE="run"
echo "DRL CONFIG: ${CONFIG_NAME}, RUN MODE: ${RUN_MODE}"

mpirun -n 1 python -m nek_MARL initial ../conf/${CONFIG_NAME} \
    > ${LOG_DIR}/log.initial.${CONFIG_NAME} 2>&1

AGENT_RUN_NAME=$(grep -ri 'agent_run_name' ../conf/${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')
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
