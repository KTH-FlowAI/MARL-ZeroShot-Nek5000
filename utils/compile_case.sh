# !/bin/bash
## Script for complie cases

case_path="./envs/nek/cases/"
case_name=mini_channel
current_path=($pwd)
version=v19
case_name_v17=small_wing

# Parse arguments for commit message
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --m) case_name="$2"; shift ;; # Set commit message
        --version) version="$2"; shift ;; # Set version
        --case_name_v17) case_name_v17="$2"; shift ;; # Set case name for v17
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

target_path=./envs/cases/${case_name}
cd ${target_path} 
echo "Complie Path $(pwd)"

if [ "$version" == "v17" ]; then
    ./compile_script clean && ./compile_script ${case_name_v17}
else
    ./compile_script --clean && ./compile_script --all 
fi

echo "[INFO] Finish Compile:  ${target_path}" 
