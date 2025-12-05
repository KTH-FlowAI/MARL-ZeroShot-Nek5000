#!/bin/bash

## Bash file for Unzip the solvers
solver_path=./envs/solver/
cd $solver_path
tar -xvf DEC_DRL_Nek5000.tar.gz   
echo "[SYS] NEK - Wing - DRL"
tar -vxf DEC_Nek5000.tar   
echo "[SYS] NEK - Wing - RAW"
tar -xvf KTH_DRL_Framework.tar.gz   
echo "[SYS] NEK - TCF - DRL"
tar -xvf KTH_Framework.tar.gz 
echo "[SYS] NEK - TCF - RAW"
