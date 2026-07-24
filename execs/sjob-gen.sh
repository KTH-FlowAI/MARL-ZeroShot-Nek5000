#!/bin/bash
# =============================================================================
#  sjob-gen.sh — SLURM job-script generator
#
#  Wraps one of the payload scripts (execs/channel-run.sh, execs/wing-run.sh)
#  into a batch script whose SLURM header — job name, partition, begin time,
#  node count, wall time — is set from the command line.
#
#  Quick start:
#      ./execs/sjob-gen.sh --case channel --config conf/MC-ng-111.yml \
#                          --mode train -J ng-111 -t 24:00:00 --submit
#
#      ./execs/sjob-gen.sh --case channel --config conf/MC-ng-111.yml \
#                          --mode evaluate --nenv 2 -J eval-ng111 --begin +3h
#
#  A channel evaluation covers runner.rank = --env-start .. --nenv, so a sweep
#  that outlives one job is submitted as a chain of jobs over disjoint ranges:
#      ... --mode evaluate --env-start 1 --nenv 4 -J eval-1-4
#      ... --mode evaluate --env-start 5 --nenv 8 -J eval-5-8
#
#      ./execs/sjob-gen.sh --case wing --config conf/NACA4412-SHAP-Vel-2540.yml \
#                          -J shap-wing -N 86 --begin 2026-06-29T16:23:42
#
#  Anything after a bare `--` is forwarded verbatim to the payload script:
#      ./execs/sjob-gen.sh --case channel --config conf/MC-ng-111.yml -- \
#                          --smpstep 12 --extra "runner.seed=7"
#
#  The generated script is written to execs/sjobs/<job-name>.sh by default and
#  can be edited by hand afterwards — it is a plain, self-contained sbatch file.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

# -----------------------------------------------------------------------------
# Defaults — the site settings that rarely change live here
# -----------------------------------------------------------------------------
CASE="channel"
CONFIGS=()
MODE=""                       # payload default when empty
ACCOUNT="deepwing"
PARTITION="batch"
TIME_LIMIT="24:00:00"
NODES="auto"                  # auto => derived from `nproc` in the config
NTASKS_PER_NODE=48
CPUS_PER_TASK=1
EXCLUSIVE="yes"
JOB_NAME=""                   # default derived from the config name
MAIL_USER="yuninw@umich.edu"
MAIL_TYPE="ALL"
BEGIN=""                      # empty => start as soon as resources allow
OUT_SCRIPT=""                 # default execs/sjobs/<job-name>.sh
SLURM_OUT=""                  # default log-files/<job-name>-%j.out
SLURM_ERR=""                  # default log-files/<job-name>-%j.err
SUBMIT="no"
PRINT_ONLY="no"
PAYLOAD_ARGS=()               # everything after `--`

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- PAYLOAD_ARGS...]

Case selection
  --case channel|wing    which payload script to wrap            [${CASE}]
  --config PATH          config file (repeatable / comma-separated)
  --mode MODE            train|run|evaluate (channel), evaluate (wing)
  --nenv N               channel evaluate: last environment of the loop
  --env-start N          channel evaluate: first environment of the loop    [1]
                         The payload runs runner.rank = env-start..nenv, so a
                         sweep too long for one job can be split over several:
                         job 1 --env-start 1 --nenv 4, job 2 --env-start 5 --nenv 8

SLURM header
  -J, --job-name NAME    #SBATCH -J                  [derived from the config]
  -p, --partition NAME   #SBATCH -p                                [${PARTITION}]
  -b, --begin WHEN       #SBATCH --begin; accepts +2h, +30m, +1d, 16:30,
                         now, tomorrow, 2026-06-29T16:23:42  [immediately]
  -t, --time HH:MM:SS    #SBATCH -t                                [${TIME_LIMIT}]
  -N, --nodes N|auto     #SBATCH -N; auto = ceil((nproc+1)/ntasks-per-node) [${NODES}]
      --ntasks-per-node N                                          [${NTASKS_PER_NODE}]
      --cpus-per-task N                                            [${CPUS_PER_TASK}]
  -A, --account NAME     #SBATCH -A                                [${ACCOUNT}]
  -e, --email ADDR       #SBATCH --mail-user                       [${MAIL_USER}]
      --mail-type TYPE   #SBATCH --mail-type                       [${MAIL_TYPE}]
      --no-exclusive     drop #SBATCH --exclusive
      --slurm-out PATH   #SBATCH --output          [log-files/<job>-%j.out]
      --slurm-err PATH   #SBATCH --error           [log-files/<job>-%j.err]

Output
  -o, --output PATH      where to write the job script  [execs/sjobs/<job>.sh]
      --submit           sbatch the script right after generating it
      --print            dump the script to stdout instead of writing a file
  -h, --help             this message
EOF
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)              CASE="$2";            shift 2 ;;
        --config)            IFS=',' read -r -a _c <<< "$2"; CONFIGS+=("${_c[@]}"); shift 2 ;;
        --mode|--run-mode)   MODE="$2";            shift 2 ;;
        --nenv)              PAYLOAD_ARGS+=(--nenv "$2"); shift 2 ;;
        --env-start|--nenv-start)
                             PAYLOAD_ARGS+=(--env-start "$2"); shift 2 ;;
        -J|--job-name)       JOB_NAME="$2";        shift 2 ;;
        -p|--partition)      PARTITION="$2";       shift 2 ;;
        -b|--begin)          BEGIN="$2";           shift 2 ;;
        -t|--time)           TIME_LIMIT="$2";      shift 2 ;;
        -N|--nodes)          NODES="$2";           shift 2 ;;
        --ntasks-per-node)   NTASKS_PER_NODE="$2"; shift 2 ;;
        --cpus-per-task)     CPUS_PER_TASK="$2";   shift 2 ;;
        -A|--account)        ACCOUNT="$2";         shift 2 ;;
        -e|--email)          MAIL_USER="$2";       shift 2 ;;
        --mail-type)         MAIL_TYPE="$2";       shift 2 ;;
        --no-exclusive)      EXCLUSIVE="no";       shift   ;;
        --slurm-out)         SLURM_OUT="$2";       shift 2 ;;
        --slurm-err)         SLURM_ERR="$2";       shift 2 ;;
        -o|--output)         OUT_SCRIPT="$2";      shift 2 ;;
        --submit)            SUBMIT="yes";         shift   ;;
        --print)             PRINT_ONLY="yes";     shift   ;;
        -h|--help)           usage; exit 0 ;;
        --)                  shift; PAYLOAD_ARGS+=("$@"); break ;;
        *) echo "[ERR] Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# -- payload script -----------------------------------------------------------
case "${CASE}" in
    channel) PAYLOAD="execs/channel-run.sh" ;;
    wing)    PAYLOAD="execs/wing-run.sh" ;;
    *) echo "[ERR] --case must be channel or wing, got: ${CASE}" >&2; exit 1 ;;
esac
[[ -x "${ROOT_DIR}/${PAYLOAD}" ]] || chmod +x "${ROOT_DIR}/${PAYLOAD}" 2>/dev/null

# -- configs ------------------------------------------------------------------
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "[ERR] --config is required" >&2; exit 1
fi
CONFIG_ABS=()
for c in "${CONFIGS[@]}"; do
    a="$(cd "${ROOT_DIR}" && realpath -e "${c}" 2>/dev/null)" || {
        echo "[ERR] Config file not found: ${c}" >&2; exit 1; }
    # Store repo-root-relative when possible: the job script cds to the root.
    CONFIG_ABS+=("${a#${ROOT_DIR}/}")
done

# -- job name -----------------------------------------------------------------
if [[ -z "${JOB_NAME}" ]]; then
    _base="$(basename "${CONFIG_ABS[0]}")"; _base="${_base%.*}"
    JOB_NAME="${MODE:+${MODE}-}${_base}"
fi
# SLURM is happy with most characters, but keep file names tame.
JOB_SLUG="$(echo "${JOB_NAME}" | tr -c 'A-Za-z0-9._-' '-' | sed 's/-\+$//')"

[[ -z "${SLURM_OUT}" ]] && SLURM_OUT="log-files/${JOB_SLUG}-%j.out"
[[ -z "${SLURM_ERR}" ]] && SLURM_ERR="log-files/${JOB_SLUG}-%j.err"
[[ -z "${OUT_SCRIPT}" ]] && OUT_SCRIPT="${ROOT_DIR}/execs/sjobs/${JOB_SLUG}.sh"

# -- begin time ---------------------------------------------------------------
# Accept the shorthands people actually type; pass anything else to SLURM as is
# (SLURM understands now+2hours, 16:30, midnight, tomorrow, ISO timestamps...).
normalize_begin() {
    local b="$1"
    [[ -z "${b}" ]] && { echo ""; return; }
    case "${b}" in
        now)                     echo "now" ;;
        +*h|+*hour|+*hours)      echo "now+$(echo "${b}" | tr -dc '0-9')hours" ;;
        +*m|+*min|+*minutes)     echo "now+$(echo "${b}" | tr -dc '0-9')minutes" ;;
        +*d|+*day|+*days)        echo "now+$(echo "${b}" | tr -dc '0-9')days" ;;
        [0-9]*h)                 echo "now+$(echo "${b}" | tr -dc '0-9')hours" ;;
        [0-9]*m)                 echo "now+$(echo "${b}" | tr -dc '0-9')minutes" ;;
        *)                       echo "${b}" ;;
    esac
}
BEGIN_NORM="$(normalize_begin "${BEGIN}")"

# -- node count ---------------------------------------------------------------
# nproc in the config counts the Nek5000 ranks; the agent adds one more.
cfg_get() { grep -ri "$2" "$1" | sed -E "s/^[^:]*:[[:space:]]*//; s/[\"']//g; s/[[:space:]]+\$//" | head -n 1; }
NPROC="$(cfg_get "${ROOT_DIR}/${CONFIG_ABS[0]}" 'nproc')"
if [[ "${NODES}" == "auto" ]]; then
    if ! [[ "${NPROC}" =~ ^[0-9]+$ ]]; then
        echo "[ERR] could not read 'nproc' from ${CONFIG_ABS[0]}; pass -N explicitly" >&2
        exit 1
    fi
    TOTAL_TASKS=$(( NPROC + 1 ))
    NODES=$(( (TOTAL_TASKS + NTASKS_PER_NODE - 1) / NTASKS_PER_NODE ))
    # A single-node job only needs the tasks it actually launches.
    [[ ${NODES} -eq 1 ]] && NTASKS_PER_NODE=${TOTAL_TASKS}
    echo "[GEN] nproc=${NPROC} (+1 agent) -> -N ${NODES} --ntasks-per-node=${NTASKS_PER_NODE}"
fi

# -----------------------------------------------------------------------------
# Compose the payload command line
# -----------------------------------------------------------------------------
PAYLOAD_CMD=()
for c in "${CONFIG_ABS[@]}"; do PAYLOAD_CMD+=(--config "${c}"); done
[[ -n "${MODE}" ]] && PAYLOAD_CMD+=(--mode "${MODE}")
PAYLOAD_CMD+=("${PAYLOAD_ARGS[@]}")

# The script path stays a ${ROOT_DIR} reference (expanded inside the job);
# every argument is quoted so values with spaces survive.
payload_line() {
    local out="\"\${ROOT_DIR}\"/${PAYLOAD}" a
    for a in "${PAYLOAD_CMD[@]}"; do out+=" $(printf '%q' "${a}")"; done
    echo "${out}"
}

# -----------------------------------------------------------------------------
# Emit the job script
# -----------------------------------------------------------------------------
SBATCH_EXCLUSIVE=""
[[ "${EXCLUSIVE}" == "yes" ]] && SBATCH_EXCLUSIVE=$'\n#SBATCH --exclusive'
SBATCH_BEGIN=""
[[ -n "${BEGIN_NORM}" ]] && SBATCH_BEGIN=$'\n#SBATCH --begin='"${BEGIN_NORM}"

read -r -d '' JOB_SCRIPT <<EOF
#!/bin/bash -l
# -----------------------------------------------------------------------------
# Generated by execs/sjob-gen.sh on $(date '+%Y-%m-%d %H:%M:%S')
# Case: ${CASE}   Config: ${CONFIG_ABS[*]}   Mode: ${MODE:-<payload default>}
# Submit from the repo root:   sbatch ${OUT_SCRIPT#${ROOT_DIR}/}
# -----------------------------------------------------------------------------

#-------- Account ------
#SBATCH -A ${ACCOUNT}

#-------- Resources ------
#SBATCH -t ${TIME_LIMIT}
#SBATCH -p ${PARTITION}${SBATCH_EXCLUSIVE}
#SBATCH -N ${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}

#-------- Identity, output and notification ------
#SBATCH -J ${JOB_NAME}
#SBATCH --mail-type=${MAIL_TYPE}
#SBATCH --mail-user=${MAIL_USER}
#SBATCH --output=${SLURM_OUT}
#SBATCH --error=${SLURM_ERR}${SBATCH_BEGIN}

# Repo root baked in at generation time, so the job does not care from which
# directory it was submitted. Override with NEK_ROOT_DIR if the tree moves.
ROOT_DIR="\${NEK_ROOT_DIR:-${ROOT_DIR}}"
cd "\${ROOT_DIR}" || exit 1
mkdir -p "\${ROOT_DIR}/log-files" "\${ROOT_DIR}/.caches"

# The payload sources the MPI/conda profiles and picks the HPC environment
# itself (it sees SLURM_JOB_ID). Keep this file free of environment logic.
$(payload_line)
EOF

if [[ "${PRINT_ONLY}" == "yes" ]]; then
    printf '%s\n' "${JOB_SCRIPT}"
    exit 0
fi

mkdir -p "$(dirname "${OUT_SCRIPT}")"
printf '%s\n' "${JOB_SCRIPT}" > "${OUT_SCRIPT}"
chmod +x "${OUT_SCRIPT}"

echo "[GEN] job script : ${OUT_SCRIPT}"
echo "[GEN] job name   : ${JOB_NAME}"
echo "[GEN] partition  : ${PARTITION}   nodes: ${NODES} x ${NTASKS_PER_NODE} tasks   time: ${TIME_LIMIT}"
echo "[GEN] begin      : ${BEGIN_NORM:-immediately}"
echo "[GEN] payload    : $(payload_line)"

if [[ "${SUBMIT}" == "yes" ]]; then
    echo "[GEN] submitting from ${ROOT_DIR}"
    ( cd "${ROOT_DIR}" && sbatch "${OUT_SCRIPT}" )
else
    echo
    echo "Submit it with:"
    echo "    cd ${ROOT_DIR} && sbatch ${OUT_SCRIPT#${ROOT_DIR}/}"
fi
