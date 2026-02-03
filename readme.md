# MARL Reinforcement Learning using NEK5000 

## Get Started
### Dependencies
+ Build your own MPI: 

        ./utils/install_mpi.sh > log.mpi 2>&1 

+ Setup your dependices of miniforge: 

        ./utils/install_miniforge.sh  > log.miniforge 2>&1 

+ Install the python env

        ./utils/install_auto_env.sh nek  > log.python 2>&1 

+ Automated modification on the package 

        source ~/.bashrc.miniforge && python ./utils/patch_supersuit.py

### Initial SetUp 
+ First set up the Solvers via: 

        ./utils/initialize_solver.sh

+ Complie the code via: 

        source ~/.bashrc.openmpi_ucx

        ./utils/compile_case.sh --m mini_channel

### Running a minimal-channel drl 
+ Run the minimal channel via: 

        cd execs && .unified-script

+ To inspect the training status: 
        
        python utils/read-history --id 1998 --mean 

+ To visualize snapshots via VISIT: 

        visit -o utils/nek_visit.NEK5000 


## **TensorBoard Logging**

Training logs are automatically saved and can be viewed with TensorBoard:
```bash
tensorboard --logdir runs/your_run_name/history/tensorboard
``` 

## **IMPORTANT MODIFICATION - AUTOMATED**
The Supersuit library requires a modification to optimize single environment usage. This can now be done automatically:
### **Manual Patching**
```bash
# Install the package first
pip install -e . --no-deps

# Apply the Supersuit patch
python utils/patch_supersuit.py
```

### **What the patch does:**
The patch modifies the `vec_env_args` function in Supersuit to avoid unnecessary environment copying when `num_envs == 1`:

```python
def vec_env_args(env, num_envs):
    if num_envs == 1:
        def env_fn():
            env_copy = env  # Direct reference for single env
            return env_copy
    else: 
        def env_fn():
            env_copy = cloudpickle.loads(cloudpickle.dumps(env))  # Deep copy for multiple envs
            return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space
```


### **Restore Original Supersuit (if needed):**
```bash
python utils/patch_supersuit.py restore
```
def vec_env_args(env, num_envs):
    def env_fn():
        env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


## The strcutures of the framework
    nek-drl/
        |-- envs/

            |-- cases # flow cases to used 
                |-- mini-channel
                |-- large-channel 
                |-- wings 

            |-- solvers # Nek5000 solvers with old and new verisons 
                |-- v17_DRL 
                |-- v17_clean 
                |-- v19_DRL
                |-- v19_clean
        
       |-- src # The source codes of the framework 
                |-- __Nek_MARL__
                |-- nek_marl.py # Environment connecting to the nek 
                |-- run.py # run the script 
                |-- evaluate.py # evaluate the run
                |-- init.py # initialization 
                |-- lib/ # Other utilities for Nek and STB3.  

       |-- utils # useful scripts for compliation and job scripts. 

       |-- runs  # The production path for cases

       |-- data  # The production path for cases

       |-- conf # The configurations used for the polices. 

       |-- execs # The folder for submitting jobs or execute the job locally
       
       |-- postprocessing # The jupyter-notebooks for post-processing
