# MC16-TD3 Configuration & MPI Interface

## 1. Configuration snapshot

### Runner block
The `runner` section locks in the reinforcement-learning hyperparameters for this TD3 experiment. It bounds every action inside `[-0.0638, 0.0638]`, runs 20 episodes with 2 warmups, and performs 2 540 environment interactions per episode while logging checkpoints every 5 episodes. TD3-specific knobs such as `policy_delay`, `tau`, `gradient_steps`, and the large `buffer_size` are set explicitly, and `rescale_actions` is enabled so the raw NEK controls fit in `[−1, 1]`. The agent keeps a reference to a policy YAML template, optionally loads a checkpoint, and normalizes inputs via `utau`. See `conf/MC16-TD3.yml` for the full values.

```3:47:conf/MC16-TD3.yml
runner: 
    ctrl_max_amp    :  0.0638       # The alpha
    ctrl_min_amp    : -0.0638      # The alpha
    action_noise    :  0.1
    npl_state       :  2
...
    policy_file     : '../conf/default_ddpg_policy.yml'
    policy          : 'rl_model_9144000_steps'
    load_agent      :  False
    normalize_input : 'utau'   # Between None, utau, std, maxmin
    rescale_actions : True
    buffer_size     : 5_000_000
    tau             : 0.005
    policy_delay    : 2
    target_policy_noise : 0.0
    target_noise_clip : 0.0
```

### Simulation block
The `simulation` section defines the NEK5000 setup: the run directory, where to find `nek5000`, the `restart_folder`, and how many MPI ranks (`nproc`) and control slots (`TOTCTRL=220`) are involved. Geometry is fixed to the `mini_channel` case derived from the listed spectral-element dimensions, and the `.par` parameters enforce fixed time steps, BDF3 integration, dealiasing, and tight convergence tolerances. CFL monitoring (`target_cfl=2.0`) and zero-net-mass-flux averaging options (`znmf_avg`) are also declared here so the Python side can diagnose instabilities before they blow up in NEK.

```52:134:conf/MC16-TD3.yml
simulation: 
    CASENAME : "phill"
    exeName : 'nek5000'
    compile_path   : "../envs/cases/mini_channel"
    restart_folder : "../data/rs6_mini_Channel"
    TOTCTRL : 220
    nproc   : 10         
    ndrl    : 6
    znmf_avg: 1
    target_cfl: 2.0
    retau : 180.0 
    y_sensing : 15.0
    lx1       :  6
    Lx        :  2.67 
    Ly        :  1.0
    Lz        :  0.8
    Nx        :  4
    Ny        :  16
    Nz        :  4
    stopAt : "numSteps"
    numSteps : 500000
    dt       : -1.0e-2
    tmax     : 100000.0
    timeStepper : "bdf3"
    variableDt : "no"
    writeControl : 'timeStep'
    writeInterval : 100_000
    dealiasing : "yes"
    filtering : "none"
    filterWeight : 0.01
    filterCutoffRatio : 0.9
    p_residualTol   :  1e-8
    v_residualTol   : 1e-8
    density         :  1.0
    viscosity       :  -2800
    advection       :  "yes"
```

## 2. Execution pipeline

The SLURM job script generator runs two binaries concurrently: first `python -m nek_MARL initial` to create `RUN_PATH.txt` and pick an agent checkpoint, and then the actual experiment is launched with `mpirun` that launches the Python driver alongside the NEK5000 binary. The second `mpirun` line lists two executables separated by `:` so MPI can establish communication between the two groups. The template is defined in `utils/sjob_gen/generate_slurm_job.py`.

```153:177:utils/sjob_gen/generate_slurm_job.py
mpirun  --mca io {params['mpi_io']} \
        -n {params['python_mpi_ranks']} python -m nek_MARL initial ../conf/$CONFIG_NAME
...
mpirun --mca pml {params['mpi_pml']} \
       --mca io {params['mpi_io']} \
       -n {params['python_mpi_ranks']} python -m nek_MARL ${RUN_MODE} ../conf/$CONFIG_NAME runner.policy=${AGENT} :\
       -n {params['nek5000_mpi_ranks']} bash -c "cd ${RUN_PATH} && ./nek5000"
```

Python ranks always include one "master" driver (rank 0) and at least one worker; the `nek_marl.parallel_env` expects to run inside MPI and uses `mpi_split` to split `MPI.COMM_WORLD` into a master color (the driver) and a worker color (the NEK-interfacing agents), then creates an intercommunicator that connects them through tag 99. See `src/lib/sb3_utils.py` for the split logic.

```25:40:src/lib/sb3_utils.py
def mpi_split(comm_world):
    mpi_rank = comm_world.Get_rank()
    mpi_size = comm_world.Get_size()
    if mpi_size < 2:
        raise RuntimeError("Requires at least 2 processes (1 Master + 1 Worker)")
    if mpi_rank == 0:
        color = 0  # Master
    else:
        color = 1  # Workers
    local_comm = comm_world.Split(color, mpi_rank)
    sub_comm = local_comm.Create_intercomm(local_leader=0, peer_comm=MPI.COMM_WORLD, 
                                            remote_leader=1, tag=99)
    return sub_comm
```

With that intercommunicator in hand, `parallel_env` uses `MPI.Info` parameters such as working directory and optional hostfile to ensure NEK spawns in the correct folder, then immediately runs `init_agent()`: the driver sends the handshake command `"INTAL"` and receives the list of NEK node IDs, GLL coordinates, and face indices so that Stable Baselines can treat every controlled GLL node as a separate agent.

```65:105:src/nek_marl.py
        mpi_info = MPI.Info.Create()
        mpi_info.Set('wdir',f"{os.getcwd()}/{self.folder}")
        mpi_info.Set('bind_to','none')
        if self.conf.simulation.hostfile != '':
            mpi_info.Set('hostfile',self.conf.simulation.hostfile)
        self.mpi_info = mpi_info
```

```167:230:src/nek_marl.py
    def init_agent(self):
        request=b"INTAL"
        self.sub_comm.Send([request,MPI.CHARACTER],dest=0,tag=tag_dict["COMMAND"]['tag'])
        ...
        self.sub_comm.Recv([node_list,MPI.INTEGER],0,tag=tag_dict['NID']['tag'])
        ...
        self.sub_comm.Recv([rank_data[k],tag_dict[k]['mpi_dtype']],nid,tag=nid+tag_dict[k]['tag'])
```

Python receives the total number of control points (`TOTCTRL`) per rank and compresses them into `agent_info`, which is later used to reshape state tensors and filter the observation buffer into the per-agent views that PettingZoo expects.

## 3. MPI data-exchange loop

Python and NEK coordinate via a small command vocabulary plus typed buffers. The MPI tags (defined in `tag_dict`) guarantee that each variable uses the correct data type (`MPI.INTEGER`, `MPI.DOUBLE`, or `MPI.CHARACTER`) and a unique numerical tag offset per NEK rank.

```804:899:src/nek_marl.py
tag_dict = {
    "NID": {"tag":1996, ... , "cate":"info"},
    ...
    'current_cfl': {"tag":1999, "mpi_dtype":MPI.DOUBLE, "cate":"request"},
    'STATE':{"tag":70000, "mpi_dtype":MPI.DOUBLE, "cate":"request"},
    'REWRD':{"tag":80000, ...},
    'ACTION':{"tag":90000, "mpi_dtype":MPI.DOUBLE, "cate":"send"},
    'COMMAND':{ "tag":22, "mpi_dtype":MPI.CHARACTER, "cate":"send"},
}
```

Each simulation step follows a strict sequence:

1. **State polling.** Python sends `STATE`, reads the current simulation time, and pulls the `NFLDC × TOTCTRL` field buffer from each relevant NEK rank. The buffers are normalized as part of `state()` before being distributed to the agents.

```430:466:src/nek_marl.py
    def state(self):
        request = b'STATE'
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])
        current_time = np.ndarray((1,),dtype=np.float64)
        self.sub_comm.Recv([current_time,MPI.DOUBLE],0,tag=1998)
        ...
        self.sub_comm.Recv([buffer,tag_dict['STATE']['mpi_dtype']], nid, tag=nid*(t+1)+tag_dict['STATE']['tag'])
```

2. **Action dispatch.** Python collects actions from the RL agent, optionally enforces the zero-net-mass-flux condition defined by `znmf_avg`, and sends one buffer per NEK rank with the same `TOTCTRL` layout. Each action message uses a tag offset of `nid + tag_dict['ACTION']['tag']`.

```468:498:src/nek_marl.py
    def action(self,ctrl_value:dict):
        request=b"CNTRL"
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])
        ...
        self.sub_comm.Send([act_buffer,tag_dict['ACTION']['mpi_dtype']],nid, tag=nid+tag_dict['ACTION']['tag'])
```

3. **Evolution.** After issuing the new controls, Python tells NEK to advance (`"EVOLV"`). NEK replies with CFL samples at every `ndrl` substep so Python can abort early if CFL exceeds `target_cfl`. Only on the final substep does NEK send back the wall-shear stresses (`"REWRD"`), which Python normalizes and logs as rewards.

```501:578:src/nek_marl.py
    def evolve(self):
        request=b"EVOLV"
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])
        ...
        current_cfl = np.ndarray((1,),dtype=np.float64)
        self.sub_comm.Recv([current_cfl,tag_dict['current_cfl']['mpi_dtype']],0,tag=tag_dict['current_cfl']['tag'])
        ...
        if i_evolv == self.conf.simulation.ndrl:
            self.sub_comm.Recv([recv_buffer,tag_dict['REWRD']['mpi_dtype']], nid, tag=nid+tag_dict["REWRD"]['tag'])
```

The normalized rewards are logged to the real-time `RewardLogger` and appended to `reward_log` for post-run analysis before being returned to the RL policy.

## 4. Additional notes

- The environment saves `current_conf.yml` inside both the run folder and `history` so collaborators can trace exactly which combination of TD3, buffer, and NEK parameters produced a given run.
- The NEK data layout (`nNID × npl_state × TOTCTRL`) is reshaped in `_distribute_field` so PettingZoo receives `(npl_state, nzs, nxs)` tensors per agent, matching the `observation_space` tied to `conf.runner.npl_state`, `nzs`, and `nxs`.
- The zero-net-mass-flux averaging scheme is enforced completely on the Python side (`avg_ZNMF`) and uses the Gauss-Lobatto-Legendre weights (`lglnodes`) so that the actions still satisfy global constraints before MPI dispatch.

## 5. NEK5000-side implementation

The Fortran driver under `envs/cases/mini_channel/drl/` is responsible for the MPI handshake, the buffer I/O, and the wall-actuation updates. At `ISTEP == 0`, `DRL_main` (`envs/cases/mini_channel/drl/drl_main.f`) runs `MPI_INTERCOMM_CREATE` to link `iglobalcomm` with the Python master (`MASTER=0`) on tag 99, then calls `drl_init` to find the wall points, build the sensing plane, and populate `TOTCTRL` entries. Once initialization finishes, all ranks enter the MPI loop: rank 0 `MPI_RECV`s fixed-length character requests from Python (`tag 22`), broadcasts them via `iglobalcomm`, and routes to the correct handler while rank > 0 just listens to the broadcast.

- `STATE` requests trigger `drl_state` → `sensing_pts_compute` → `drl_state_out` (see `envs/cases/mini_channel/drl/drl_state.f` and `drl_IO.f`). The state writer sends the current time (`tag 1998`), CFL (`tag 1999`), and each field slice (`tag = NID*il + 70000`) so Python can reassemble the `fld` buffer exactly as expected by `tag_dict`.
- `CNTRL` requests invoke `drl_action`, which calls `recv_Actions` in `envs/cases/mini_channel/drl/drl_action.f`. NEK receives action buffers via `MPI_RECV(..., tag=NID+90000)`, writes them into the `ACTIONS` common block, applies the ZNMF averaging or weighted masks, and stores the results before the next PDE advance.
- `EVOLV` flips the `evolving` flag to `.true.` so the solver runs `drl_step` substeps. After reaching the final substep, NEK calls `drl_reward` followed by `drl_reward_out` to send back the CFL (`tag 1999`) and the per-node wall-shear stress reward vector (`tag = NID + 80000`), matching Python’s expectation of a reward message once per interaction.
- `TERMN` triggers `stop_simulation` which prints diagnostics and calls `exitt0`, ensuring the MPI intercommunicator and underlying NEK binary exit cleanly alongside Python’s teardown.
- During the initial handshake, `drl_info_out` gathers `NUMCTRL`, `info_agt`, and `pos_agt` arrays and sends them over tags that align with Python’s `tag_dict` offsets (`NID + 10000*k` for integer info, `NID + 100000*k` for coordinates). This lets Python know how many agents exist on each rank and how to reshape the rewards and observations later.

In short, the NEK Fortran code drives the time stepping and enforces the physical constraints, while the Python driver acts as a command-and-control layer that requests state snapshots, sends actions over typed MPI buffers, and interprets the returned CFL/reward scalars.
