#!/bin/bash
# =============================================================================
#  pol_selftest.sh -- offline validation of the embedded F77 policy
#
#  Exports an SB3 checkpoint to the .pol format, then replays the reference
#  table through drl/pol_core.f and reports how far the Fortran actor is from
#  Stable-Baselines3's own predict(). No Nek5000 and no MPI are involved.
#
#      ./utils/pol_selftest.sh --ckpt runs/mc_nes_nek/logs/best_model.zip \
#                              --config conf/mini_channel/MC-nes.yml
#
#  --skip-export reuses the .pol/.chk already in the build directory.
#
#  The build directory defaults to ~/.cache/nek_pol_selftest because /tmp is
#  commonly mounted noexec on the compute nodes here, which would refuse to
#  run the freshly linked test binary.
# =============================================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"

CKPT="runs/mc_nes_nek/logs/best_model.zip"
CONFIG="conf/mini_channel/MC-nes.yml"
CASE="mini_channel"
BUILD="${HOME}/.cache/nek_pol_selftest"
NCHK=20000
SKIP_EXPORT="no"

usage() {
    sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)        CKPT="$2";   shift 2 ;;
        --config)      CONFIG="$2"; shift 2 ;;
        --case)        CASE="$2";   shift 2 ;;
        --build)       BUILD="$2";  shift 2 ;;
        --nchk)        NCHK="$2";   shift 2 ;;
        --skip-export) SKIP_EXPORT="yes"; shift ;;
        -h|--help)     usage ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

CORE="${ROOT_DIR}/envs/cases/${CASE}/drl/pol_core.f"
DRV="${ROOT_DIR}/utils/pol_selftest.f"
NAME="$(basename "${CKPT%.*}")"

mkdir -p "${BUILD}" || exit 1

if [[ "${SKIP_EXPORT}" == "no" ]]; then
    ORIG_ARGS=("$@"); set --
    source ~/.bashrc.miniforge
    set -- "${ORIG_ARGS[@]}"

    echo "[SELFTEST] exporting ${CKPT}"
    ( cd "${ROOT_DIR}" && python3 utils/sb3_to_f77.py "${CKPT}" \
        --config "${CONFIG}" --nchk "${NCHK}" \
        --out "${BUILD}/${NAME}.pol" ) || exit 1
fi

echo "[SELFTEST] building ${CORE##*/} + ${DRV##*/}"
gfortran -O2 -std=legacy -o "${BUILD}/pol_selftest" "${DRV}" "${CORE}" || exit 1

cd "${BUILD}" || exit 1
./pol_selftest "${NAME}.pol" "${NAME}.chk" "${NAME}.bits"
