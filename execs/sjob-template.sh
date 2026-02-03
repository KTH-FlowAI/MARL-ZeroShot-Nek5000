#!/bin/bash -l
#### A Template for submitting your JOB

#-------- Account ------
#SBATCH -A <account>

#-------- Partions ------------
#SBATCH -t <time>
#SBATCH -p <partition>
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH --ntasks-per-node=<ntasks-per-node>
#SBATCH --cpus-per-task=1

#-------- Outputs and notification ------
#SBATCH -J <job-name>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email>
#SBATCH --output=<output-file>
#SBATCH --error=<error-file>

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
CONFIG_NAME="MC16-TD3.yml"
RUN_MODE="run"
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
#--------------------------------------------

echo "DRL CONFIG: ${CONFIG_NAME}, RUN MODE: ${RUN_MODE}"
mpirun  --mca io ompio \
        -n 1 python -m nek_MARL initial ../conf/${CONFIG_NAME} > ${LOG_DIR}/log.initial.${CONFIG_NAME} 2>&1 

## Identify the path-to-go
AGENT_RUN_NAME=$(grep -ri 'agent_run_name' ../conf/${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')
NTOT=$(grep -ri 'nproc' ../conf/${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')
RUN_PATH=$(head -n 1 RUN_PATH_${AGENT_RUN_NAME}.txt)

if [ "$RUN_MODE" = "run" ]; then
    AGENT=$(tail -n 1 RUN_PATH_${AGENT_RUN_NAME}.txt)
else
    AGENT=""
fi

##--- History management
if [ "$RUN_MODE" = "run" ]; then
    fileNUM=$(ls "${RUN_PATH}"/round* -dq | wc -l)
    fileNUM=$((fileNUM + 1))
    printf -v fileNUM "%03d" "$fileNUM"  # Zero-pad to 3 digits
    HISTORY_PATH="${RUN_PATH}/round${fileNUM}"
    mv "${RUN_PATH}/history" "${HISTORY_PATH}"
    echo "MV HISTORY: ${HISTORY_PATH}"
fi

##--- RUN! To use mpi_split, one should launch two programs at once. Therefore 
if [ "$RUN_MODE" = "evaluate" ]; then
    echo "EVALUATE MODE"
    for ienv in {1..6}
    do
    mpirun --mca pml ucx \
           --mca io ompio \
           -n 1 python -m nek_MARL ${RUN_MODE} ../conf/${CONFIG_NAME} \
           runner.rank=${ienv} runner.random_init=-1 :\
           runner.evaluation=True :\
           runner.learnt_policy=True :\
           runner.load_agent=True :\
           -n ${NTOT} bash -c "cd ${RUN_PATH} && ./nek5000" > ${LOG_DIR}/log.evaluate.${CONFIG_NAME}.${ienv} 2>&1 
    done
else
    echo "RUN MODE"
    mpirun --mca pml ucx \
       --mca io ompio \
       -n 1 python -m nek_MARL ${RUN_MODE} ../conf/${CONFIG_NAME} runner.policy=${AGENT} :\
       -n ${NTOT} bash -c "cd ${RUN_PATH} && ./nek5000" > ${LOG_DIR}/log.run.${CONFIG_NAME} 2>&1 
fi
