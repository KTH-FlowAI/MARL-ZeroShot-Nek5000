#!/bin/bash -l
#### A Template for submitting your JOB

#-------- Account ------
#SBATCH -A deepwing

#-------- Partions ------------
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1

#-------- Outputs and notification ------
#SBATCH -J DDPG-MINI
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuninw@umich.edu
#SBATCH --output=./log-files/log-ddpg-mini
#SBATCH --error=./log-files/log-ddpg-mini

# Environment 
#------------------------
source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
unset MPI_UCX_ROOT
unset UCX_ROOT
export UCX_WARN_UNUSED_ENV_VARS=n
echo "Environment ALL SET"
#------------------------


# Summary
#------------------------
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
#------------------------

CONFIG_NAME="MC16-DDPG.yml"

RUN_MODE="run"
echo "DRL CONFIG: ${CONFIG_NAME}, RUN MODE: ${RUN_MODE}"
mpirun  --mca io ompio \
        -n 1 python -m nek_MARL initial conf/$CONFIG_NAME
## Identify the path-to-go
RUN_PATH=$(head -n 1 RUN_PATH.txt)
AGENT=$(tail -n 1 RUN_PATH.txt)
## Sort the log history
if [ "$RUN_MODE" = "run" ]; then
    fileNUM=$(ls "${RUN_PATH}"/round* -dq | wc -l)
    fileNUM=$((fileNUM + 1))
    printf -v fileNUM "%03d" "$fileNUM"  # Zero-pad to 3 digits
    HISTORY_PATH="${RUN_PATH}/round${fileNUM}"
    mv "${RUN_PATH}/history" "${HISTORY_PATH}"
    echo "MV HISTORY: ${HISTORY_PATH}"
fi

## RUN! To use mpi_split, one should launch two programs at once. Therefore 
mpirun --mca pml ucx \
       --mca io ompio \
       -n 1 python -m nek_MARL ${RUN_MODE} conf/$CONFIG_NAME runner.policy=${AGENT} :\
       -n 32 bash -c "cd ${RUN_PATH} && ./nek5000"
