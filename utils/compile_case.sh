#!/bin/bash
## Script for complie cases
##
## Usage:
##   ./utils/compile_case.sh --path envs/cases/mini_channel [--solver drl|raw]
##
## --solver drl  (default) build against KTH_DRL_Framework -> ./nek5000
##               Coupled mode: rank 0 belongs to Python.
## --solver raw            build against KTH_Framework     -> ./nek5000_solo
##               Embedded mode: stock core, UPARAM(10)=1.
##
## Both binaries live in the same case folder and survive each other's
## builds. Switching solvers is a from-scratch rebuild, which costs
## nothing extra here because this script always cleans first anyway.

case_path=mini_channel
solver=""

# Parse arguments for commit message
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --path)   case_path="$2"; shift ;; # Set commit message
        --m)      case_path="$2"; shift ;;
        --solver) solver="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

target_path=$(realpath ${case_path})
cd ${target_path} || exit 1
echo "[INFO] Complie $(pwd)"

SOLVER_ARG=()
if [[ -n "${solver}" ]]; then
    SOLVER_ARG=(--solver "${solver}")
    echo "[INFO] Solver backend: ${solver}"
fi

#[MOD] makenek asks "cleanup all 3rd party dependencies too? [N]" on the
#[MOD] clean path, and asks again if it detects a configuration change.
#[MOD] Either would hang a batch job, so answer them from a pipe instead
#[MOD] of relying on somebody being at the keyboard. 'N' keeps the
#[MOD] prebuilt gslib/blasLapack, which live under each solver root and
#[MOD] are therefore never shared between the two builds.
printf 'N\nN\n' | ./compile_script "${SOLVER_ARG[@]}" --clean \
  && printf 'N\nN\n' | ./compile_script "${SOLVER_ARG[@]}" --all

echo "[INFO] Finish Compile:  ${target_path}"
