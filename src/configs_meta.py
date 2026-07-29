"""
Configuration of runner and NEK for meta-evaluation (wing case)
@yuningw
"""
from dataclasses import dataclass, field
import argparse
from omegaconf import OmegaConf
from typing import Optional, List, Any
import time


@dataclass
class Runner:
    """
    Runner only for evaluation!
    """

    # Essential Parameter
    nb_interactions: int = 3000  # Number of "interactions" for stepping schedules

    # Agent loading/resuming options
    random_init: int = 2  # -1==No Shuffle, use the No.init = RANK; -2==NOT Cover the current rs8

    # The Meta Running case
    case_name: str = ""

    # A list of lists of control area
    agent_ctrl_area: list = field(default_factory=list)
    # A list of lists of control side "SS" or "PS", indicating the suction side or pressure side
    agent_ctrl_side: list = field(default_factory=list)
    # A list of integer to indicate agent case
    agent_run_name: list = field(default_factory=list)
    # A list of string to indicate which policy to read
    policy: list = field(default_factory=list)
    # A list of string to indicate the RL algorithm
    RL_algorithm: list = field(default_factory=list)
    # A list of float indicating the local u_tau
    u_tau: list = field(default_factory=list)
    # A list of float indicating the local dU/dy
    dUdy: list = field(default_factory=list)
    # Action bounds in list
    action_bounds: list = field(default_factory=list)
    # Action update in list
    drl_steps: list = field(default_factory=list)
    # A list of string to indicate the solver souce for transfer learning
    source_solvers: list = field(default_factory=list) # now we only support "dedalus" and "nek", 
    # but this can be easily extended to more solvers in the future

    evaluation: bool = True
    rank: int = 0  # NAME of test case
    learnt_policy: bool = True

    vars_record: bool = True
    vars_record_freq: int = 1
    vars_io_freq: int = 100  # Frequency to save the data
    netgain_io_freq: int = 200  # Flush per-chord-point NETGAIN time-series every N control steps

    # Normalizing Interaction
    rew_mode: str = 'Homo'  # We let all subdomain share the same reward
    normalize_input: str = 'None'  # Between None, utau, std
    size_history: int = 10

    # Action Related
    ctrl_array_size: int = 1
    # Do not rescale action as we need to adjust this locally
    rescale_actions: bool = False
    # It is no longer used since we scale the actions locally
    ctrl_min_amp: float = -1.0
    ctrl_max_amp: float = 1.0

    # u and v
    npl_state: int = 2  # Should be consistent with SIZE

    reward_fn:str       = 'dudy'  # 'dudy' or 'net_gain' (requires NETGAIN compile flag in Fortran)
    reward_alpha:float  = 1.0    # weight on R_wallshear = 1 - tau_w/tau_w_ref
    reward_beta:float   = 1.0    # weight on R_pw        = -|p'v|/tau_w_ref
    reward_gamma:float  = 1.0    # weight on R_v3        = -0.5|v^3|/tau_w_ref


@dataclass
class Simulation:
    CASENAME: str = "naca_wing"
    solver_version: str = "v17"
    # I/O
    exeName: str = 'nek5000'
    nzs: int = 1
    nxs: int = 1
    out_log: str = 'logfile'
    host: str = 'localhost'
    hostfile: str = ''
    compile_path: str = '01_compile'
    restart_folder: str = 'restart-files'
    #[MOD] Shared, case-independent dependencies (mesh, mask, ...) that are too
    #[MOD] bulky to duplicate in every envs/cases/<case> folder. Files not found
    #[MOD] in compile_path are looked up here, e.g. "data/simulations/naca4412_75k".
    #[MOD] Accepts a single path or a list of paths ('' / [] disables it).
    shared_data_path: Any = ''
    #[MOD] Symlink (instead of copy) the files resolved from shared_data_path.
    shared_data_link: bool = True
    # VERY IMPORTANT
    # --------------------------------
    nproc: int = 14  # ranks for running
    TOTCTRL: int = 10  # Be consistent with SIZE
    ndrl: int = 3
    znmf_avg: int = 1  # 1==Open
    target_cfl: float = 0.5
    # --------------------------------

    # slurm specific
    num_nodes_srun: int = 4
    oversubscribe: bool = False
    # ------------------------
    # SIZE
    # ------------------------
    lx1: int = 6  # Polynomial Order
    Lx: float = 3.141
    Ly: float = 2.0
    Lz: float = 2.0  # Size of domain
    Nx: int = 4
    Ny: int = 16
    Nz: int = 4  # No. Spectral Elements
    tmax: float = 1400
    # --------------------------
    # .rea file
    # --------------------------
    # General
    numSteps: int = 10  # p011
    dt: float = -1.0e-05  # p012
    target_cfl: float = 0.4  # p026
    # Write Interval of La2 and Reg
    writeInterval: int = 5000  # p016
    writeLA2: int = 5000  # p070
    # PRESSURE
    p_residualTol: float = 1e-8  # p021
    v_residualTol: float = 1e-8  # p022
    density: float = 1.0  # p001
    viscosity: float = -0.75E5  # p002

    # _CHKPOINT - checkpoint module
    READCHKPT: int = 1  # p076 # Restart from checkpoint
    CHKPFNUMBER: int = 6  # p067 # Restart file number
    CHKPINTERVAL: int = 5000  # p075 # Checkpoint saving frequency (time steps)

    # _STAT - statistics module
    AVSTEP: int = 10  # p087
    IOSTEP: int = 10000  # p088

    # _TSRS
    writePTS: int = 50


# Parameter mapping (NEK v17)
param_mapping = {
    "density": "p001",
    "viscosity": "p002",
    "numSteps": "p011",
    "dt": "p012",
    "writeInterval": "p015",
    "target_cfl": "p026",
    "writeLA2": "p070",
    "p_residualTol": "p021",
    "v_residualTol": "p022",
    "writePTS": "p051",
    "READCHKPT": "p076",
    "CHKPFNUMBER": "p067",
    "CHKPINTERVAL": "p075",
    "AVSTEP": "p087",
    "IOSTEP": "p088",
    "ndrl": "p089",
    "znmf_avg": "p090",
}


@dataclass
class Embedded:
    """#[MOD] Python-free evaluation for the meta (multi-region) stack.

    The region table is taken from the Runner section that MetaPolicy
    already uses -- agent_ctrl_area / agent_ctrl_side / u_tau /
    action_bounds / drl_steps / source_solvers -- so a wing config needs
    nothing beyond `enabled: True`. The fields here only exist to
    override that on a per-region basis.
    """
    enabled:bool        = False
    net_precision:int   = 8       # accumulation precision: 8 (default) or 4
    rec_freq:int        = 1       # record every N control cycles
    rec_bufsize:int     = 100     # records buffered before a flush
    # Opt-in: use the newest complete local solver checkpoint on preparation.
    resume:bool          = False
    # Also write binary drlrec files when Python drives the coupled solver.
    coupled_recorder:bool = False
    # Optional classical law written directly as a .pol (currently OC or BL).
    analytic_policy:Optional[str] = None

    ctrl_areas:Any      = None    # [[xmin, xmax], ...]
    ctrl_sides:Any      = None    # ['ANY' | 'SS' | 'PS', ...]
    policies:Any        = None    # explicit checkpoint paths
    nupd:Any            = None    # interactions between updates
    u_tau:Any           = None
    ctrl_max_amp:Any    = None


@dataclass
class Logging:
    run_name: str = str(int(time.time()))  # str so it can hold a string agent_run_name
    group: Optional[str] = None
    notes: Optional[str] = None
    save_dir: str = 'runs'
    policy_dir: str = "runs" # Policy folder is the same as the run folder, this is an adation to the current structure


@dataclass
class Config:
    simulation: Simulation = Simulation()
    runner: Runner = Runner()
    logging: Logging = Logging()
    embedded: Embedded = Embedded()      #[MOD] Python-free evaluation


def add_subparser(parser: argparse.ArgumentParser):
    subparser = parser.add_parser("conf", help="Configuration")

    subparser.add_argument(
        "files_or_overrides",
        type=str,
        metavar="arg",
        nargs="*",
        help="Config YAML files e.g. `conf.yaml` or overrides e.g. `other.gpus=4`",
    )
    subparser.set_defaults(cmd=parse_cli)


def parse_cli(files_or_overrides: List[str], **ignored_kwargs):
    files = []
    overrides = []
    for x in files_or_overrides:
        if "=" in x:
            overrides.append(x)
        elif x.endswith((".yaml", ".yml")):
            files.append(x)
        else:
            raise ValueError(f"Unrecognized: {x}")
    conf = OmegaConf.merge(
        OmegaConf.structured(Config()),
        *map(OmegaConf.load, files),
        OmegaConf.from_dotlist(overrides),
    )
    print(OmegaConf.to_yaml(conf, resolve=True))


if __name__ == '__main__':
    pass
