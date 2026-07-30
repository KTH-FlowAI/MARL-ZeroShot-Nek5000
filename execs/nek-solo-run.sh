#!/bin/bash
# =============================================================================
#  nek-solo-run.sh
#
#  Launch an EMBEDDED (Python-free) evaluation: Nek5000 evaluates the actor
#  itself, so there is no Python rank, no MPI_Comm_spawn and no per-cycle
#  message exchange. See temp/docs/embedded_policy.md.
#
#      ./execs/nek-solo-run.sh --config conf/mini_channel/MC-nes.yml
#      ./execs/nek-solo-run.sh --config conf/mini_channel/MC-nes.yml --nenv 4
#
#  It carries no SLURM directives, so the same command runs interactively and
#  inside a batch job. Wrap it with the job generator to submit it.
#
#  What it does, per env rank:
#    1. `python -m nek_MARL|meta_MARL initial ... embedded.enabled=True`
#         stages the case, writes the .par with userParam10 = 1, exports
#         actor_NNN.pol, writes drl_policy.in and creates drlrec/.
#    2. `mpirun -n <nproc> ./nek5000_solo`
#         plain Nek. Every rank runs the solver -- unlike the coupled mode,
#         rank 0 is NOT reserved for Python.
#
#  REQUIREMENTS
#    * the case must be built against the RAW solver:
#         ./utils/compile_case.sh --path envs/cases/<case> --solver raw
#      which produces ./nek5000_solo alongside the coupled ./nek5000.
#    * a checkpoint to fly, resolved from runner.agent_run_name/runner.policy
#      unless embedded.policies is given explicitly.
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

set -o pipefail

# Keep a generous stack for solver code outside the reward path. The large
# reward work arrays are static common-block storage now, so this is no longer
# a correctness mitigation for embedded wing runs.
ulimit -s unlimited 2>/dev/null || ulimit -s 65536 2>/dev/null || true
export OMP_STACKSIZE="${OMP_STACKSIZE:-256M}"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
CONFIGS=()
SITE=""                    # local|hpc  (empty => auto-detect)
ENV_START=1
NENV=1
EXENAME="nek5000_solo"
RANDOM_INIT=""             # empty => honour runner.random_init in the YAML
NB_INTERACTIONS=""
REC_FREQ=""
PRECISION=""
MPI_OPTS=""
EXTRA_OVERRIDES=""
MV_DATA="no"                 # archive a previous env before preparation
CASE_TAG=""                   # archive directory; defaults to simulation.CASENAME
DATA_ID=""                    # archive env id; defaults to the solo env rank
DRY_RUN="no"
SKIP_PREPARE="no"
RESUME="no"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

  --config PATH        config YAML, relative to the repo root or absolute.
                       Repeatable, or comma-separated.
  --nenv N             last runner.rank of the loop            [${NENV}]
  --env-start N        first runner.rank of the loop           [${ENV_START}]
  --exe NAME           solver binary to run                    [${EXENAME}]
  --nb-interactions N  override runner.nb_interactions
  --rec-freq N         override embedded.rec_freq
  --precision 4|8      override embedded.net_precision         [config]
  --random-init N      override runner.random_init             [config value]
  --resume              continue from the newest complete local checkpoint
  --site local|hpc     environment flavour; auto-detected from SLURM_JOB_ID
  --mpi-opts "FLAGS"   replace the site default mpirun flags
  --mv-data yes|no     archive the previous environment first              [${MV_DATA}]
  --case-name NAME     archive directory under data/results/                [simulation.CASENAME]
  --id NNN             archive env id (single-environment run only)        [env rank]
  --extra "K=V K=V"    extra config overrides for the prepare step
  --skip-prepare       reuse the run folder as it stands (no re-export)
  --dry-run            print the commands without running them
  -h, --help           this message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)          IFS=',' read -ra _c <<< "$2"; CONFIGS+=("${_c[@]}"); shift 2 ;;
        --nenv)            NENV="$2"; shift 2 ;;
        --env-start)       ENV_START="$2"; shift 2 ;;
        --exe)             EXENAME="$2"; shift 2 ;;
        --nb-interactions) NB_INTERACTIONS="$2"; shift 2 ;;
        --rec-freq)        REC_FREQ="$2"; shift 2 ;;
        --precision)       PRECISION="$2"; shift 2 ;;
        --random-init)     RANDOM_INIT="$2"; shift 2 ;;
        --site)            SITE="$2"; shift 2 ;;
        --mpi-opts)        MPI_OPTS="$2"; shift 2 ;;
        --mv-data)         MV_DATA="$2"; shift 2 ;;
        --case-name)       CASE_TAG="$2"; shift 2 ;;
        --id)              DATA_ID="$2"; shift 2 ;;
        --extra)           EXTRA_OVERRIDES="$2"; shift 2 ;;
        --skip-prepare)    SKIP_PREPARE="yes"; shift ;;
        --resume)          RESUME="yes"; shift ;;
        --dry-run)         DRY_RUN="yes"; shift ;;
        -h|--help)         usage ;;
        *) echo "[ERR] unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "[ERR] give at least one --config" >&2; exit 1; }
if [[ "${RESUME}" == "yes" && "${SKIP_PREPARE}" == "yes" ]]; then
    echo "[ERR] --resume needs preparation to select its checkpoint; omit --skip-prepare" >&2
    exit 1
fi
case "${MV_DATA}" in
    yes|no) ;;
    *) echo "[ERR] --mv-data must be yes or no, got: ${MV_DATA}" >&2; exit 1 ;;
esac
if [[ -n "${DATA_ID}" && ! "${DATA_ID}" =~ ^[0-9][0-9][0-9]$ ]]; then
    echo "[ERR] --id must be a three-digit environment id, got: ${DATA_ID}" >&2
    exit 1
fi
if [[ -n "${DATA_ID}" && "${ENV_START}" != "${NENV}" ]]; then
    echo "[ERR] --id is only valid for one environment; omit it to archive each env rank" >&2
    exit 1
fi

# Site detection, matching channel-run.sh.
if [[ -z "${SITE}" ]]; then
    if [[ -n "${SLURM_JOB_ID}" ]]; then SITE="hpc"; else SITE="local"; fi
fi
case "${SITE}" in
    local) MPI_RUN_OPTS="" ;;
    hpc)   export UCX_WARN_UNUSED_ENV_VARS=n
           MPI_RUN_OPTS="--mca io ompio" ;;
    *) echo "[ERR] --site must be local or hpc, got: ${SITE}" >&2; exit 1 ;;
esac
[[ -n "${MPI_OPTS}" ]] && MPI_RUN_OPTS="${MPI_OPTS}"

cd "${ROOT_DIR}" || exit 1
LOG_DIR="${ROOT_DIR}/log-files"
CACHE_DIR="${ROOT_DIR}/.caches"
mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

# -----------------------------------------------------------------------------
# Helpers (same semantics as channel-run.sh, kept independent on purpose so
# the two launchers cannot break each other)
# -----------------------------------------------------------------------------
cfg_get() {
    awk -v k="$2" '
        BEGIN { re = "^[[:space:]]*" k "[[:space:]]*:" }
        /^[[:space:]]*#/ { next }
        $0 ~ re {
            sub(/^[^:]*:[[:space:]]*/, "")
            sub(/[[:space:]]*#.*$/, "")
            gsub(/["'"'"']/, "")
            sub(/[[:space:]]+$/, "")
            print; exit
        }' "$1"
}

read_run_path() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        if [[ "${DRY_RUN}" == "yes" ]]; then
            RUN_PATH="<not written yet: ${file}>"
            echo "[DRY] cache file absent: ${file}"; return 0
        fi
        echo "[ERR] \`initial\` did not write the cache file: ${file}" >&2
        return 1
    fi
    RUN_PATH="$(sed -n 1p "${file}")"
    [[ -n "${RUN_PATH}" ]] || { echo "[ERR] empty run path in ${file}" >&2; return 1; }
    return 0
}

run_cmd() {
    local log="$1"; shift
    if [[ "${DRY_RUN}" == "yes" ]]; then
        printf '[DRY] %s\n[DRY]   > %s\n' "$*" "${log}"; return 0
    fi
    if [[ "${log}" == "-" ]]; then eval "$@"; else eval "$@" > "${log}" 2>&1; fi
}

echo "============================================================"
echo " nek-solo-run  |  site: ${SITE}  |  embedded (no Python rank)"
echo " root        : ${ROOT_DIR}"
echo " configs     : ${CONFIGS[*]}"
echo " env ranks   : ${ENV_START}..${NENV}"
echo " solver      : ${EXENAME}"
[[ -n "${SLURM_JOB_ID}" ]] && echo " slurm job   : ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo " started at  : $(date)"
echo "============================================================"

# -----------------------------------------------------------------------------
for _cfg_in in "${CONFIGS[@]}"; do
    CONFIG_NAME="$(realpath -e "${_cfg_in}" 2>/dev/null)" || {
        echo "[ERR] config file not found: ${_cfg_in}" >&2; exit 1; }
    CONFIG_TAG="$(basename "${CONFIG_NAME}")"
    NTOT="$(cfg_get "${CONFIG_NAME}" 'nproc')"
    AGENT_RUN_NAME="$(cfg_get "${CONFIG_NAME}" 'agent_run_name')"
    CASE_NAME="$(cfg_get "${CONFIG_NAME}" 'case_name')"
    CONFIG_CASE_TAG="$(cfg_get "${CONFIG_NAME}" 'CASENAME')"
    COMPILE_PATH="$(cfg_get "${CONFIG_NAME}" 'compile_path')"

    # The channel stack writes RUN_PATH_<agent_run_name>.txt and is entered
    # through nek_MARL.  The multi-region wing stack writes
    # RUN_PATH_<case_name>.txt through meta_MARL.  `case_name` is exclusive to
    # the latter, so it is a stable discriminator without adding a launcher
    # flag that could drift from the YAML.
    if [[ -n "${CASE_NAME}" ]]; then
        MARL_MODULE="meta_MARL"
        RUN_KEY="${CASE_NAME}"
    else
        MARL_MODULE="nek_MARL"
        RUN_KEY="${AGENT_RUN_NAME}"
    fi
    ARCHIVE_CASE="${CASE_TAG:-${CONFIG_CASE_TAG}}"

    echo "------------------------------------------------------------"
    echo "[CFG] ${CONFIG_NAME}"
    echo "[CFG] module=${MARL_MODULE}  run_key=${RUN_KEY}  nproc=${NTOT}"
    [[ "${MV_DATA}" == "yes" ]] && echo "[CFG] archive=${ARCHIVE_CASE} (per env rank)"

    [[ -n "${RUN_KEY}" ]] || {
        echo "[ERR] no runner.agent_run_name or runner.case_name in ${CONFIG_NAME}" >&2; exit 1; }
    [[ "${NTOT}" =~ ^[0-9]+$ ]] || {
        echo "[ERR] simulation.nproc is not a number: '${NTOT}'" >&2; exit 1; }
    if [[ "${MV_DATA}" == "yes" && -z "${ARCHIVE_CASE}" ]]; then
        echo "[ERR] --mv-data needs simulation.CASENAME or --case-name" >&2
        exit 1
    fi

    # The embedded mode NEEDS the raw-solver binary. Catch it here rather than
    # letting the run hang in MPI_INTERCOMM_CREATE waiting for a Python rank
    # that is never going to connect.
    if [[ -n "${COMPILE_PATH}" && "${DRY_RUN}" == "no" ]]; then
        if [[ ! -x "${ROOT_DIR}/${COMPILE_PATH}/${EXENAME}" ]]; then
            echo "[ERR] ${COMPILE_PATH}/${EXENAME} not found or not executable." >&2
            echo "[ERR] Build it with:" >&2
            echo "[ERR]   ./utils/compile_case.sh --path ${COMPILE_PATH} --solver raw" >&2
            exit 1
        fi
    fi

    EMB_ARGS="embedded.enabled=True simulation.exeName=${EXENAME}"
    [[ -n "${NB_INTERACTIONS}" ]] && EMB_ARGS+=" runner.nb_interactions=${NB_INTERACTIONS}"
    [[ -n "${REC_FREQ}" ]]        && EMB_ARGS+=" embedded.rec_freq=${REC_FREQ}"
    [[ -n "${PRECISION}" ]]       && EMB_ARGS+=" embedded.net_precision=${PRECISION}"
    [[ "${RESUME}" == "yes" ]]    && EMB_ARGS+=" embedded.resume=True"

    for ienv in $(seq "${ENV_START}" "${NENV}"); do
        if [[ -n "${DATA_ID}" ]]; then
            ARCHIVE_ID="${DATA_ID}"
        else
            printf -v ARCHIVE_ID '%03d' "${ienv}"
        fi
        INIT_LOG="${LOG_DIR}/log.solo-initial.${CONFIG_TAG}.env_${ARCHIVE_ID}"
        SOLO_LOG="${LOG_DIR}/log.solo.${CONFIG_TAG}.env_${ARCHIVE_ID}"
        # Earlier solo launches used one log per config. Retain that log as a
        # fallback when archiving the first run after this per-env convention.
        ARCHIVE_LOG="${SOLO_LOG}"
        [[ -f "${ARCHIVE_LOG}" ]] || ARCHIVE_LOG="${LOG_DIR}/log.solo.${CONFIG_TAG}"

        echo "[SOLO] rank ${ienv} of ${ENV_START}..${NENV} (env_${ARCHIVE_ID})"
        # The archive intentionally precedes `initial`: embedded preparation
        # regenerates actor_*.pol and drl_policy.in, so archiving later would
        # lose the previous controller provenance even though solver fields
        # have not yet been overwritten.
        if [[ "${MV_DATA}" == "yes" ]]; then
            run_cmd "-" "LOGFILE='${ARCHIVE_LOG}' \
                bash '${ROOT_DIR}/utils/mv-data' --source_root '${ROOT_DIR}' \
                --case_name '${ARCHIVE_CASE}' --run_name '${RUN_KEY}' \
                --id '${ARCHIVE_ID}'" || {
                echo "[ERR] archive failed for ${RUN_KEY}/env_${ARCHIVE_ID}" >&2
                exit 1; }
        fi
        INIT_ARGS="runner.rank=${ienv} runner.evaluation=True "
        if [[ "${MARL_MODULE}" == "meta_MARL" ]]; then
            INIT_ARGS+="runner.learnt_policy=True"
        else
            INIT_ARGS+="runner.learnt_policy=True runner.load_agent=True"
        fi
        [[ -n "${RANDOM_INIT}" ]] && INIT_ARGS+=" runner.random_init=${RANDOM_INIT}"

        if [[ "${SKIP_PREPARE}" == "no" ]]; then
            run_cmd "${INIT_LOG}" \
                "mpirun -n 1 python -m ${MARL_MODULE} initial ${CONFIG_NAME} \
                    ${INIT_ARGS} ${EMB_ARGS} ${EXTRA_OVERRIDES}" || {
                echo "[ERR] prepare failed, see ${LOG_DIR}/log.solo-initial.${CONFIG_TAG}" >&2
                exit 1; }
        fi

        read_run_path "${CACHE_DIR}/RUN_PATH_${RUN_KEY}.txt" || exit 1
        echo "[SOLO] RUN_PATH=${RUN_PATH}"

        # No Python rank: every one of the NTOT ranks runs Nek.
        run_cmd "${SOLO_LOG}" \
            "mpirun ${MPI_RUN_OPTS} -n ${NTOT} \
                bash -c 'cd ${RUN_PATH} && ./${EXENAME}'" || {
            echo "[ERR] solver failed, see ${SOLO_LOG}" >&2
            exit 1; }

        if [[ "${DRY_RUN}" == "no" ]]; then
            NREC="$(ls "${ROOT_DIR}/${RUN_PATH}/drlrec"/*.bin 2>/dev/null | wc -l)"
            echo "[SOLO] rank ${ienv} done, ${NREC} record files in ${RUN_PATH}/drlrec"
        fi
    done
done

echo "============================================================"
echo " finished at : $(date)"
echo "============================================================"
