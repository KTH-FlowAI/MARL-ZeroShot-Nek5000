# Source code

This folder contains the code to train and evaluate the MARL agents:
- **configs.py**: contains the default configuration of all the parameters
- **run.py** and **evaluate.py**: contain the routines to train and evaluate the agents, respectively. 
- **nek_marl.py**: contains the definition of the environment using the PettingZoo/Gym framework for DRL.
- **lib/**: contains additional routines to generate the solver input files, for instance.

A script for evaluation is also provided
- **evaluate-script.sh**: allows to run several evaluations at the same time. The evaluation runs are saved in the `runs/[timestamp]/` folder.  

