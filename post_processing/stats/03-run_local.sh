#!/bin/bash 

rm log01.txt

casename=phill
rm  -f $casename.sch
echo $casename > SESSION.NAME
echo $PWD/ >> SESSION.NAME

mpirun -n 4 ./nek5000 | tee log01.txt
