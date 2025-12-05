# !/bin/bash
## Script for complie cases

case_path="./envs/nek/cases/"
case_name=mini_channel
current_path=($pwd)

# Parse arguments for commit message
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --m) case_name="$2"; shift ;; # Set commit message
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

target_path=./envs/cases/${case_name}
cd ${target_path} 
echo "Complie Path $(pwd)"

./compile_script --clean && ./compile_script --all 

echo "[INFO] Finish Compile:  ${target_path}" 
