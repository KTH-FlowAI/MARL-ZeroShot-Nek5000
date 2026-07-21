#!/bin/bash
# =============================================================================
#  channel-run.sh
#
#  Common execution commands for the MINIMAL-CHANNEL cases (python -m nek_MARL).
#
#  This script holds *only* the payload: environment, config resolution and the
#  mpirun invocations. It carries no SLURM directives, so the very same command
#  runs interactively on a login/workstation node and inside a batch job:
#
#      ./execs/channel-run.sh --config conf/mini_channel/MC-ng-111.yml --mode train
#      ./execs/channel-run.sh --config conf/mini_channel/MC-ng-111.yml --mode evaluate --nenv 2
#
#  To submit it to SLURM, wrap it with the job generator:
#
#      ./execs/sjob-gen.sh --case channel --config conf/mini_channel/MC-ng-111.yml \
#                          --mode train -J my-run --begin +2h --submit
#
#  The site (local vs. HPC) is auto-detected from $SLURM_JOB_ID; override with
#  --site if needed.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

# Shield the script arguments from the sourced shell profiles: they forward
# "$@" to `activate`, which errors when it receives more than one argument.
ORIG_ARGS=("$@")
set --
source ~/.bashrc.openmpi_ucx
source ~/.bashrc.miniforge
set -- "${ORIG_ARGS[@]}"

#[MOD] Set only after the profiles are sourced, so a failing pipeline inside
#[MOD] them cannot abort the launcher.
set -o pipefail

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
CONFIGS=()                 # --config may be repeated / comma-separated
MODE="train"               # train|run|evaluate
SITE=""                    # local|hpc   (empty => auto-detect)
LOAD_AGENT=""              # ""|True|False  (empty keeps the config value)
NENV=2                     # evaluate: loop over runner.rank = 1..NENV
IOSTEP=5000
WRITE_INTERVAL=""          # empty => follow IOSTEP
SMPSTEP=6
REWARD_FN="net_gain"       # evaluate: forced so all reward components are logged
REWARD_ALPHA=1.0
REWARD_BETA=1.0
REWARD_GAMMA=1.0
RANDOM_INIT=-1             # evaluate only
NB_INTERACTIONS=""         # evaluate only; empty keeps the config value
MPI_OPTS=""                # extra mpirun flags, overrides the site defaults
EXTRA_OVERRIDES=""         # free-form "key=value key=value" hydra-style overrides
DRY_RUN="no"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

  --config PATH        config file, relative to the repo root or absolute.
                       Repeatable, or comma-separated, to sweep several cases.
  --mode MODE          train|run (training loop) or evaluate  [${MODE}]
  --site local|hpc     environment flavour; auto-detected from SLURM_JOB_ID
  --load-agent True|False
                       override runner.load_agent; empty keeps the config value
  --nenv N             evaluate: run runner.rank = 1..N        [${NENV}]
  --iostep N           simulation.IOSTEP                       [${IOSTEP}]
  --write-interval N   simulation.writeInterval          [follows --iostep]
  --smpstep N          simulation.SMPSTEP                      [${SMPSTEP}]
  --reward-fn NAME     evaluate: runner.reward_fn              [${REWARD_FN}]
  --alpha/--beta/--gamma VALUE
                       evaluate: reward weights      [${REWARD_ALPHA}/${REWARD_BETA}/${REWARD_GAMMA}]
  --random-init N      evaluate: runner.random_init            [${RANDOM_INIT}]
  --nb-interactions N  evaluate: runner.nb_interactions (statistics runs)
  --mpi-opts "FLAGS"   replace the site default mpirun flags (e.g. "--mca pml ucx")
  --extra "K=V K=V"    extra overrides appended to every python call
  --dry-run            print the commands instead of running them
  -h, --help           this message

Examples:
  # training
  $(basename "$0") --config conf/mini_channel/MC-ng-111.yml --mode train

  # evaluation of a trained policy over two environments
  $(basename "$0") --config conf/mini_channel/MC-ng-111.yml --mode evaluate --nenv 2 \\
                   --iostep 10000 --write-interval 10000 --smpstep 12

  # long statistics run over a set of cases
  $(basename "$0") --mode evaluate --nenv 1 --nb-interactions 20000 \\
                   --config conf/mini_channel/MC-shapcf.yml --config conf/mini_channel/MC-shapvel.yml
EOF
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           IFS=',' read -r -a _c <<< "$2"; CONFIGS+=("${_c[@]}"); shift 2 ;;
        --mode|--run-mode)  MODE="$2";            shift 2 ;;
        --site)             SITE="$2";            shift 2 ;;
        --load-agent)       LOAD_AGENT="$2";      shift 2 ;;
        --nenv)             NENV="$2";            shift 2 ;;
        --iostep)           IOSTEP="$2";          shift 2 ;;
        --write-interval)   WRITE_INTERVAL="$2";  shift 2 ;;
        --smpstep)          SMPSTEP="$2";         shift 2 ;;
        --reward-fn)        REWARD_FN="$2";       shift 2 ;;
        --alpha)            REWARD_ALPHA="$2";    shift 2 ;;
        --beta)             REWARD_BETA="$2";     shift 2 ;;
        --gamma)            REWARD_GAMMA="$2";    shift 2 ;;
        --random-init)      RANDOM_INIT="$2";     shift 2 ;;
        --nb-interactions)  NB_INTERACTIONS="$2"; shift 2 ;;
        --mpi-opts)         MPI_OPTS="$2";        shift 2 ;;
        --extra)            EXTRA_OVERRIDES="$2"; shift 2 ;;
        --dry-run)          DRY_RUN="yes";        shift   ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "[ERR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

#[MOD] The old default (conf/MC16-TD3.yml) no longer exists in the repo.
[[ ${#CONFIGS[@]} -eq 0 ]] && CONFIGS=("conf/mini_channel/MC-ng-111.yml")
# Writing the fields more often than the checkpoints is never what is wanted,
# so writeInterval tracks IOSTEP unless it is given explicitly.
[[ -z "${WRITE_INTERVAL}" ]] && WRITE_INTERVAL="${IOSTEP}"

case "${MODE}" in
    train|run) RUN_MODE="run" ;;
    evaluate)  RUN_MODE="evaluate" ;;
    *) echo "[ERR] --mode must be train|run|evaluate, got: ${MODE}" >&2; exit 1 ;;
esac

if [[ -n "${LOAD_AGENT}" && "${LOAD_AGENT}" != "True" && "${LOAD_AGENT}" != "False" ]]; then
    echo "[ERR] --load-agent must be True or False, got: ${LOAD_AGENT}" >&2; exit 1
fi
LOAD_AGENT_ARG=""
[[ -n "${LOAD_AGENT}" ]] && LOAD_AGENT_ARG="runner.load_agent=${LOAD_AGENT}"

NB_INTER_ARG=""
[[ -n "${NB_INTERACTIONS}" ]] && NB_INTER_ARG="runner.nb_interactions=${NB_INTERACTIONS}"

# -----------------------------------------------------------------------------
# Site environment
#   local : oversubscribed shared-memory run on a workstation/login node
#   hpc   : inside a SLURM allocation, UCX over the fabric
# -----------------------------------------------------------------------------
if [[ -z "${SITE}" ]]; then
    SITE="local"; [[ -n "${SLURM_JOB_ID}" ]] && SITE="hpc"
fi

export HWLOC_HIDE_ERRORS=1
case "${SITE}" in
    local)
        unset OMPI_MCA_pml OMPI_MCA_osc UCX_TLS UCX_NET_DEVICES
        export UCX_TLS=sm,self,tcp,cma,sysv,posix
        export OMPI_MCA_btl=self,vader,tcp
        MPI_RUN_OPTS=""
        MPI_EVAL_OPTS=""
        ;;
    hpc)
        export UCX_WARN_UNUSED_ENV_VARS=n
        MPI_RUN_OPTS="--mca io ompio"
        MPI_EVAL_OPTS="--mca pml ucx"
        ;;
    *) echo "[ERR] --site must be local or hpc, got: ${SITE}" >&2; exit 1 ;;
esac
# An explicit --mpi-opts wins over the site defaults for every mpirun call.
if [[ -n "${MPI_OPTS}" ]]; then
    MPI_RUN_OPTS="${MPI_OPTS}"; MPI_EVAL_OPTS="${MPI_OPTS}"
fi

# Run python from the repo root so that conf/... and save_dir=runs resolve here.
cd "${ROOT_DIR}" || exit 1
LOG_DIR="${ROOT_DIR}/log-files"
CACHE_DIR="${ROOT_DIR}/.caches"     # RUN_PATH_*.txt written by `initial`
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
# Pull a scalar out of a yml config (first match wins, quotes stripped).
#[MOD] The key must start the line, comment lines are skipped and a trailing
#[MOD] "# ..." is cut off. The previous unanchored `grep` returned the first
#[MOD] line that merely MENTIONED the key -- e.g. the
#[MOD] "## [REMINDER] ... the agent_run_name prefix is NOT auto-added ..."
#[MOD] comment sitting above the key in some configs -- which produced a
#[MOD] non-existent RUN_PATH_*.txt, an empty RUN_PATH and a bare `cd` (== $HOME)
#[MOD] for every solver rank.
cfg_get() {
    awk -v k="$2" '
        BEGIN { re = "^[[:space:]]*" k "[[:space:]]*:" }
        /^[[:space:]]*#/ { next }
        $0 ~ re {
            sub(/^[^:]*:[[:space:]]*/, "")   # drop the key
            sub(/[[:space:]]*#.*$/, "")      # drop a trailing comment
            gsub(/["'"'"']/, "")             # drop quotes
            sub(/[[:space:]]+$/, "")         # drop trailing blanks
            print; exit
        }' "$1"
}

# read_run_path <cache-file>  ->  sets RUN_PATH (line 1) and AGENT (line 2)
#[MOD] `initial` writes the run folder on line 1 and the latest checkpoint stem
#[MOD] on line 2 (empty on a fresh run). `tail -n 1` used to fall back to line 1
#[MOD] when line 2 was absent, so `run` was handed the run folder as its policy
#[MOD] and RLA.load() died on a nonsense path.
read_run_path() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        if [[ "${DRY_RUN}" == "yes" ]]; then
            RUN_PATH="<not written yet: ${file}>"; AGENT=""
            echo "[DRY] cache file absent: ${file}"
            return 0
        fi
        echo "[ERR] \`initial\` did not write the cache file: ${file}" >&2
        return 1
    fi
    RUN_PATH="$(sed -n 1p "${file}")"
    AGENT="$(sed -n 2p "${file}")"
    if [[ -z "${RUN_PATH}" ]]; then
        echo "[ERR] Empty run path in ${file}" >&2
        return 1
    fi
    return 0
}

# run_cmd <logfile|-> <command ...>   ('-' keeps the output on the terminal)
run_cmd() {
    local log="$1"; shift
    if [[ "${DRY_RUN}" == "yes" ]]; then
        printf '[DRY] %s\n[DRY]   > %s\n' "$*" "${log}"
        return 0
    fi
    if [[ "${log}" == "-" ]]; then eval "$@"; else eval "$@" > "${log}" 2>&1; fi
}

echo "============================================================"
echo " channel-run  |  site: ${SITE}  |  mode: ${RUN_MODE}"
echo " root        : ${ROOT_DIR}"
echo " configs     : ${CONFIGS[*]}"
[[ -n "${SLURM_JOB_ID}" ]] && echo " slurm job   : ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo " started at  : $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
# Main loop over the requested configs
# -----------------------------------------------------------------------------
for _cfg_in in "${CONFIGS[@]}"; do

    CONFIG_NAME="$(realpath -e "${_cfg_in}" 2>/dev/null)" || {
        echo "[ERR] Config file not found: ${_cfg_in}" >&2; exit 1; }
    CONFIG_TAG="$(basename "${CONFIG_NAME}")"
    NTOT="$(cfg_get "${CONFIG_NAME}" 'nproc')"
    AGENT_RUN_NAME="$(cfg_get "${CONFIG_NAME}" 'agent_run_name')"

    echo "------------------------------------------------------------"
    echo "[CFG] ${CONFIG_NAME}"
    echo "[CFG] agent_run_name=${AGENT_RUN_NAME}  nproc=${NTOT}"

    #[MOD] Both values index files/rank counts further down, so a missing or
    #[MOD] non-numeric read must stop the job instead of producing `mpirun -n`
    #[MOD] garbage or a RUN_PATH_.txt lookup.
    [[ -n "${AGENT_RUN_NAME}" ]] || {
        echo "[ERR] runner.agent_run_name not found in ${CONFIG_NAME}" >&2; exit 1; }
    [[ "${NTOT}" =~ ^[0-9]+$ ]] || {
        echo "[ERR] simulation.nproc is not a number in ${CONFIG_NAME}: '${NTOT}'" >&2; exit 1; }
    #[MOD] The shell reads the yml directly while python reads yml+overrides, so
    #[MOD] overriding either of these two in --extra would desync the launcher
    #[MOD] from the solver (wrong cache file / wrong rank count -> MPI hang).
    case "${EXTRA_OVERRIDES}" in
        *runner.agent_run_name=*|*simulation.nproc=*|*logging.save_dir=*)
            echo "[ERR] --extra must not override agent_run_name / nproc /" \
                 "save_dir; edit the config instead: ${EXTRA_OVERRIDES}" >&2; exit 1 ;;
    esac

    if [[ "${RUN_MODE}" == "run" ]]; then
        # -- Training -------------------------------------------------------
        # History archiving/cleanup happens inside `initial`
        # (src/initial.py:preserve_and_clean_train); nothing to move here.
        #[MOD] A failed `initial` must abort: the cache file below would other-
        #[MOD] wise still hold the PREVIOUS run's path and the solver would
        #[MOD] happily start in the wrong folder.
        run_cmd "${LOG_DIR}/log.initial.${CONFIG_TAG}" \
            "mpirun -n 1 python -m nek_MARL initial ${CONFIG_NAME} ${EXTRA_OVERRIDES}" || {
            echo "[ERR] initial failed, see ${LOG_DIR}/log.initial.${CONFIG_TAG}" >&2; exit 1; }

        read_run_path "${CACHE_DIR}/RUN_PATH_${AGENT_RUN_NAME}.txt" || exit 1
        echo "[RUN] RUN_PATH=${RUN_PATH}  policy=${AGENT:-<none, training from scratch>}"

        #[MOD] No checkpoint found by `initial` => nothing to resume from. A
        #[MOD] config that says load_agent: True would otherwise send SB3 into
        #[MOD] RLA.load() with an empty policy name. An explicit --load-agent
        #[MOD] still wins, so the failure stays visible when it is asked for.
        RUN_LOAD_AGENT_ARG="${LOAD_AGENT_ARG}"
        if [[ -z "${AGENT}" && -z "${LOAD_AGENT}" ]]; then
            RUN_LOAD_AGENT_ARG="runner.load_agent=False"
            echo "[RUN] no checkpoint yet -> runner.load_agent=False"
        fi

        if [[ "${SITE}" == "local" ]]; then
            # One communicator, rank 0 drives the agent: what oversubscribed
            # workstation runs have been using.
            run_cmd "${LOG_DIR}/log.run.${CONFIG_TAG}" \
                "mpirun -n \$((1 + ${NTOT})) bash -c '
                    if [ \$OMPI_COMM_WORLD_RANK -eq 0 ]; then
                        python -m nek_MARL run ${CONFIG_NAME} runner.policy=${AGENT} ${RUN_LOAD_AGENT_ARG} ${EXTRA_OVERRIDES}
                    else
                        cd ${RUN_PATH} && ./nek5000
                    fi'"
        else
            run_cmd "${LOG_DIR}/log.run.${CONFIG_TAG}" \
                "mpirun ${MPI_RUN_OPTS} \
                    -n 1 python -m nek_MARL run ${CONFIG_NAME} \
                        runner.policy=${AGENT} ${RUN_LOAD_AGENT_ARG} ${EXTRA_OVERRIDES} : \
                    -n ${NTOT} bash -c 'cd ${RUN_PATH} && ./nek5000'"
        fi

    else
        # -- Evaluation -----------------------------------------------------
        # reward_fn drives userParam09, i.e. the Fortran 1-vs-3 MPI buffer
        # count. It MUST be passed identically to `initial` (which writes the
        # .par) and to `evaluate` below — forcing only one side desyncs the
        # buffer count and deadlocks MPI.
        REWARD_ARGS="runner.reward_fn=${REWARD_FN} \
            runner.reward_alpha=${REWARD_ALPHA} \
            runner.reward_beta=${REWARD_BETA} \
            runner.reward_gamma=${REWARD_GAMMA}"
        SIM_ARGS="simulation.IOSTEP=${IOSTEP} \
            simulation.writeInterval=${WRITE_INTERVAL} \
            simulation.SMPSTEP=${SMPSTEP}"

        for ienv in $(seq 1 "${NENV}"); do
            echo "[EVAL] rank ${ienv}/${NENV}"

            run_cmd "${LOG_DIR}/log.initial.${CONFIG_TAG}" \
                "mpirun -n 1 python -m nek_MARL initial ${CONFIG_NAME} \
                    runner.random_init=${RANDOM_INIT} \
                    runner.rank=${ienv} \
                    ${REWARD_ARGS} \
                    runner.evaluation=True runner.learnt_policy=True \
                    runner.load_agent=True \
                    ${SIM_ARGS} ${EXTRA_OVERRIDES}" || {
                echo "[ERR] initial failed, see ${LOG_DIR}/log.initial.${CONFIG_TAG}" >&2; exit 1; }

            read_run_path "${CACHE_DIR}/RUN_PATH_${AGENT_RUN_NAME}.txt" || exit 1
            echo "[EVAL] RUN_PATH=${RUN_PATH}"

            run_cmd "${LOG_DIR}/log.eval.${CONFIG_TAG}" \
                "mpirun ${MPI_EVAL_OPTS} \
                    -n 1 python -m nek_MARL evaluate ${CONFIG_NAME} \
                        runner.rank=${ienv} runner.random_init=${RANDOM_INIT} \
                        ${REWARD_ARGS} ${NB_INTER_ARG} \
                        runner.evaluation=True \
                        runner.load_agent=True \
                        ${SIM_ARGS} ${EXTRA_OVERRIDES} : \
                    -n ${NTOT} bash -c 'cd ${RUN_PATH} && ./nek5000'"
        done
    fi
done

echo "============================================================"
echo " finished at : $(date)"
echo "============================================================"
