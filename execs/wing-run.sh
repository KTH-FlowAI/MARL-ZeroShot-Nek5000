#!/bin/bash
# =============================================================================
#  wing-run.sh
#
#  Common execution commands for the NACA4412 WING cases (python -m meta_MARL).
#
#  Payload only: no SLURM directives, so the same command line works on a
#  workstation and inside a batch job:
#
#      ./execs/wing-run.sh --config conf/NACA4412-SHAP-Vel-2540.yml --mode evaluate
#
#  To submit it to SLURM, wrap it with the job generator:
#
#      ./execs/sjob-gen.sh --case wing --config conf/NACA4412-SHAP-Vel-2540.yml \
#                          -J shap-wing -N 86 --begin +2h --submit
#
#  The site (local vs. HPC) is auto-detected from $SLURM_JOB_ID.
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
CONFIGS=()
MODE="evaluate"            # meta_MARL currently only supports evaluate
SITE=""                    # local|hpc (empty => auto-detect)
MV_DATA="no"               # archive the previous results via utils/mv-data
CASE_TAG="naca_wing"       # --case-name passed to utils/mv-data
DATA_ID="001"              # env_<ID> folder picked up by utils/mv-data
MPI_OPTS=""
EXTRA_OVERRIDES=""
DRY_RUN="no"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

  --config PATH        config file, relative to the repo root or absolute.
                       Repeatable, or comma-separated.
  --mode MODE          evaluate                                [${MODE}]
  --site local|hpc     environment flavour; auto-detected from SLURM_JOB_ID
  --mv-data yes|no     archive results with utils/mv-data first [${MV_DATA}]
  --case-name NAME     case folder used by utils/mv-data        [${CASE_TAG}]
  --id NNN             env id used by utils/mv-data             [${DATA_ID}]
  --mpi-opts "FLAGS"   replace the site default mpirun flags
  --extra "K=V K=V"    extra overrides appended to every python call
  --dry-run            print the commands instead of running them
  -h, --help           this message

Examples:
  $(basename "$0") --config conf/WING-SMALL.yml
  $(basename "$0") --config conf/NACA4412-SHAP-Vel-2540.yml --mv-data yes
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
        --mv-data)          MV_DATA="$2";         shift 2 ;;
        --case-name)        CASE_TAG="$2";        shift 2 ;;
        --id)               DATA_ID="$2";         shift 2 ;;
        --mpi-opts)         MPI_OPTS="$2";        shift 2 ;;
        --extra)            EXTRA_OVERRIDES="$2"; shift 2 ;;
        --dry-run)          DRY_RUN="yes";        shift   ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "[ERR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ ${#CONFIGS[@]} -eq 0 ]] && CONFIGS=("conf/WING-SMALL.yml")

if [[ "${MODE}" != "evaluate" ]]; then
    echo "[ERR] RUN MODE NOT SUPPORTED for the wing case: ${MODE}" >&2; exit 1
fi

# -----------------------------------------------------------------------------
# Site environment
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
        MPI_INIT_OPTS=""
        MPI_EVAL_OPTS=""
        ;;
    hpc)
        unset MPI_UCX_ROOT UCX_ROOT
        export UCX_WARN_UNUSED_ENV_VARS=n
        MPI_INIT_OPTS="--mca io ompio"
        MPI_EVAL_OPTS="--mca pml ucx --mca io ompio"
        ;;
    *) echo "[ERR] --site must be local or hpc, got: ${SITE}" >&2; exit 1 ;;
esac
if [[ -n "${MPI_OPTS}" ]]; then
    MPI_INIT_OPTS="${MPI_OPTS}"; MPI_EVAL_OPTS="${MPI_OPTS}"
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
#[MOD] line that merely MENTIONED the key (e.g. a "## [REMINDER] ..." comment),
#[MOD] which produced a non-existent RUN_PATH_*.txt, an empty RUN_PATH and a
#[MOD] bare `cd` (== $HOME) for every solver rank.
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

# read_run_path <cache-file>  ->  sets RUN_PATH (line 1 of the cache file)
#[MOD] Guarded so a failed `initial` can no longer leave RUN_PATH empty, which
#[MOD] turned the solver launch into a bare `cd` into $HOME.
read_run_path() {
    local file="$1"
    if [[ ! -f "${file}" ]]; then
        if [[ "${DRY_RUN}" == "yes" ]]; then
            RUN_PATH="<not written yet: ${file}>"
            echo "[DRY] cache file absent: ${file}"
            return 0
        fi
        echo "[ERR] \`initial\` did not write the cache file: ${file}" >&2
        return 1
    fi
    RUN_PATH="$(sed -n 1p "${file}")"
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
echo " wing-run  |  site: ${SITE}  |  mode: ${MODE}"
echo " root       : ${ROOT_DIR}"
echo " configs    : ${CONFIGS[*]}"
[[ -n "${SLURM_JOB_ID}" ]] && echo " slurm job  : ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo " started at : $(date)"
echo "============================================================"

for _cfg_in in "${CONFIGS[@]}"; do

    CONFIG_NAME="$(realpath -e "${_cfg_in}" 2>/dev/null)" || {
        echo "[ERR] Config file not found: ${_cfg_in}" >&2; exit 1; }
    CONFIG_TAG="$(basename "${CONFIG_NAME}")"

    echo "------------------------------------------------------------"
    echo "[CFG] ${CONFIG_NAME}"

    #[MOD] A failed `initial` must abort: the cache file below would otherwise
    #[MOD] still hold the PREVIOUS run's path and the solver would happily start
    #[MOD] in the wrong folder.
    run_cmd "${LOG_DIR}/log.initial.${CONFIG_TAG}" \
        "mpirun ${MPI_INIT_OPTS} -n 1 python -m meta_MARL initial ${CONFIG_NAME} ${EXTRA_OVERRIDES}" || {
        echo "[ERR] initial failed, see ${LOG_DIR}/log.initial.${CONFIG_TAG}" >&2; exit 1; }

    CASE_NAME="$(cfg_get "${CONFIG_NAME}" 'case_name')"
    NTOT="$(cfg_get "${CONFIG_NAME}" 'nproc')"
    echo "[CFG] case_name=${CASE_NAME}  nproc=${NTOT}"

    #[MOD] Both index files/rank counts below, so a bad read must stop the job.
    [[ -n "${CASE_NAME}" ]] || {
        echo "[ERR] runner.case_name not found in ${CONFIG_NAME}" >&2; exit 1; }
    [[ "${NTOT}" =~ ^[0-9]+$ ]] || {
        echo "[ERR] simulation.nproc is not a number in ${CONFIG_NAME}: '${NTOT}'" >&2; exit 1; }
    #[MOD] The shell reads the yml directly while python reads yml+overrides, so
    #[MOD] overriding either of these in --extra would desync the launcher from
    #[MOD] the solver (wrong cache file / wrong rank count -> MPI hang).
    case "${EXTRA_OVERRIDES}" in
        *runner.case_name=*|*simulation.nproc=*|*logging.save_dir=*)
            echo "[ERR] --extra must not override case_name / nproc / save_dir;" \
                 "edit the config instead: ${EXTRA_OVERRIDES}" >&2; exit 1 ;;
    esac

    read_run_path "${CACHE_DIR}/RUN_PATH_${CASE_NAME}.txt" || exit 1
    echo "[RUN] RUN_PATH=${RUN_PATH}"

    # Archive whatever the previous run left in runs/<case>/env_<id> before the
    # solver starts overwriting it. Run it in a child shell (not sourced): the
    # helper assigns CASE_NAME/RUN_PATH itself and would clobber ours. LOGFILE
    # is the previous evaluation log, which it copies next to the data.
    if [[ "${MV_DATA}" == "yes" ]]; then
        run_cmd "-" "LOGFILE='${LOG_DIR}/log.eval.${CONFIG_TAG}' \
            bash ${ROOT_DIR}/utils/mv-data --source_root ${ROOT_DIR} \
            --case_name ${CASE_TAG} --run_name ${CASE_NAME} --id ${DATA_ID}"
    fi

    # mpi_split needs both programs launched by the same mpirun.
    run_cmd "${LOG_DIR}/log.eval.${CONFIG_TAG}" \
        "mpirun ${MPI_EVAL_OPTS} \
            -n 1 python -m meta_MARL evaluate ${CONFIG_NAME} ${EXTRA_OVERRIDES} : \
            -n ${NTOT} bash -c 'cd ${RUN_PATH} && ./nek5000'"
done

echo "============================================================"
echo " finished at : $(date)"
echo "============================================================"
