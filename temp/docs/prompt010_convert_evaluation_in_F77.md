# Converting Deterministic Reactive Control schemes into Purely Nek5000 codes

Working path `/home/yuninw/codes/drl/1_Nek/nek_power_saving`
## Background 
This repo is for DRL for drag-reduction control for turbulent wing and channel. For more details please read `/home/yuninw/codes/drl/1_Nek/nek_power_saving/readme.md` and the documentations in ./temp/docs/ for previous updates.

The idea of using MPI to bridge Python & F77 Nek5000 is a great idea which enables the exploration doable. However, when the problem scales up to a turbulent wings or even high-Re Channel, the time consumption using MPI for deterministic control evaluation is a bit too much as the control parameters are *Deterministic*. And the actor network essentially is an 8-neuron shallow MLP (`conf/default_ddpg_policy.yml`).  

This makes me thinking of using a pure, complied F77 programe to carry out the evaluation instead. Because MLP at the end of the day requires a very minimal operation for matrix multiplication, which in the Fortran generally has marginal costs.

## Objective
### Program
I would like you to design a program in F77 worked with NEK5000, which realizes the following workflow: 
1. Reads the Weight & Bias of the deterministic Actor networks (in double precision) at the initialization stage. 
2. Perform the operations that mirrors what MLP does to the partial observation. Note there the input is just a 2-dimension vector (u', v').
3. Impose the actutaion as the boundary condition.
4. Calculate the defined rewards on the fly.
5. The actuation and the obervation should also be written and dumped on the fly. This can be realized by the time-series `TSRS` or a self-defined binary outputs. Actually I preferred the latter as it will not violate the TSRS interpolation especially for the wing. 
6. It should also supports multiple policies collaborating, like the function as `META_WING`
7. The `ndrl` update interval should be fully respected. 
8. The F77 subroutines that are drl-related should be fully respected. 
9. The design need to be fully documented and user-friendly (to certain extend XD)

### Tips
1. Although this program should be modulized and ready for plug and play. But please note this is F77 not F90. I would recommend you to create several scripts with an common block to share the tensors in memory. See drl dependencies in `envs/cases/mini_channel/drl/` and its asscociated common block `/home/yuninw/codes/drl/1_Nek/nek_power_saving/envs/cases/mini_channel/inc_src/DRL`.   
2. The framework logic is exactly the same, now we are just avioding the MPI communications. The source ./src/ implementation should give you sufficient information
3. The MPI usage 

```
source ~/.bashrc.openmpi_ucx
```
4. Python env:

```
source ~/.bashrc.miniforge
```


### Useful toolkits to be created:
1. A reader that unpack the checkpoint data from StableBaselines3 
2. A writer in F77 to record (action, observation and reward) on the fly and a A Reader of Python to unpack it. 
3. A new sets of job submition scripts which exectues Nek5000 only.  

### Developing Process. 
+ I would like to start with a minimal channel case `envs/cases/mini_channel/drl/`. 
+ For the reference W&B of actor I recommend you look at `runs/mc_nes_nek/logs/best_model.zip ` and you can use the associated configs. 
Steps
1. Inspect the codes in the repo carefully and evaluate the feasibility. 
2. Start with the minimal channel case and test with that. 
3. The target is to obtain the exactly same drag reduction using the new code. 


# Follow-up-1
Thanks, my respones are: 
1. Regarding the precision, thanks for bring it up. I agree with that and please keep float32 for W&B 
2. Note that the loading objective is not necessarily always be `best_model.zip`, so we need some degree of freendom here in the configuration.
3. Can you clarify: Will you fully rely on the `.par` or `.rea` for the parameter or we are using indenpendent configuration class like the `.yml` used here?
4. Follow-up on the configuration: Can we manage the parameters like `u_tau` `action_max/min`, `nb_interations` and also the recording frequency? 
5. The reward function should use the net-energy saving instead. 
6. There is some locks connected to DRL usage in the source NEK5000 main code which I modified (`./envs/solver/DEC_DRL_Nek5000`) and (`./envs/solver/KTH_DRL_Framework`) see `/core/` folder find the main program and check if there is anything related. 
Please discuss with my comments for the moment.

# Follow-up-2
1. Regarding the solver backend, we also have the origin solver codes (`./envs/solver/DEC_Nek5000`) and (`./envs/solver/KTH_Framework`) which does not have this python lock. Can you inspect them and get back to me? Also can you find another descrepancy? 

2. I am happy with the configuration design. Just wanted you to clarify it so I understand fully.

3. The rests looks good.