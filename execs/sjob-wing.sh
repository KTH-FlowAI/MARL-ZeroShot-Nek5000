#!/bin/bash -l
#### A Template for submitting your JOB

#-------- Account ------
#SBATCH -A deepwing

#-------- Partions ------------
#SBATCH -t 24:00:00
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -N 86
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1

#-------- Outputs and notification ------
#SBATCH -J Fede-Wing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuninw@umich.edu
#SBATCH --output=log-files/log.wing
#SBATCH --error=log-files/log.wing
#SBATCH --begin=2026-06-03T04:22:37
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
CONFIG_NAME="NACA4412-SHAP-Vel-2540.yml"
RUN_MODE="evaluate"
MV_DATA="yes"
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
        --mv-data)
            MV_DATA="$2"
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
        -n 1 python -m meta_MARL initial ../conf/${CONFIG_NAME} 

## Identify the path-to-go
CASE_NAME=$(grep -ri 'case_name' ../conf/${CONFIG_NAME} | sed -E "s/^[^:]*:[[:space:]]*//; s/[\"']//g; s/[[:space:]]+\$//")
NTOT=$(grep -ri 'nproc' ../conf/${CONFIG_NAME} | awk -F':' '{gsub(/ /,"",$2); print $2}')
RUN_PATH=$(head -n 1 RUN_PATH_${CASE_NAME}.txt)

#----- sort out the data -----
if [ "$MV_DATA" = "yes" ]; then
    . ../utils/mv-data --source_root .. --case_name naca_wing --run_name $CASE_NAME  --id 001  
fi
#----------------------------

##--- RUN! To use mpi_split, one should launch two programs at once.  
#--------------------------------------------
# Main execution
if [ "$RUN_MODE" = "evaluate" ]; then
    # Evaluation mode execution
    mpirun --mca pml ucx \
       --mca io ompio \
       -n 1 python -m meta_MARL  evaluate ../conf/$CONFIG_NAME :\
       -n ${NTOT} bash -c "cd ${RUN_PATH} && ./nek5000" > ${LOG_DIR}/log.evaluate.${CONFIG_NAME} 2>&1 
else
    echo "RUN MODE NOT SUPPORTED"
    exit 1
fi

