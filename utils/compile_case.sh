# !/bin/bash
## Script for complie cases

case_path=mini_channel
# Parse arguments for commit message
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --path) case_path="$2"; shift ;; # Set commit message
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

target_path=$(realpath ${case_path})
cd ${target_path} 
echo "[INFO] Complie $(pwd)"

./compile_script --clean && ./compile_script --all 

echo "[INFO] Finish Compile:  ${target_path}" 
