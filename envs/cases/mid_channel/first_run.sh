#!/bin/bash

#SBATCH -J chan180
#SBATCH -A snic2022-3-25


#SBATCH -p main

#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -n 128
#SBATCH --ntasks-per-node=128

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=marco.atzori.92@gmail.com

rm *.sch

module swap PrgEnv-cray/8.2.0 PrgEnv-intel/8.2.0
#module load cray-mpich/8.1.11

srun -n 128 ./nek5000 >> logRun_T011.txt 2>&1

FOLDER=Run_011_T
mkdir $FOLDER
mv phill*.f* $FOLDER
cp rs6phill0.f00004 rs6phill0.f00005 rs6phill0.f00006 $FOLDER
mv c2Dphill0.f0000* stsphill0.f0000* $FOLDER


