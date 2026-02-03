#!/bin/bash 

## Auto-detecting installation script for any conda environment
## Usage: ./install_auto_env.sh [environment_name]

#---- Configuration ----#
TARGET_BASH=~/.bashrc.miniforge
MINIFORGE_ABS=$(realpath ./miniforge3)
source ~/.bashrc   # Renew the conda installation
source ~/.bashrc.openmpi_ucx

# Get environment name from argument or auto-detect
if [[ "$1" != "" ]]; then
    ENV_NAME="$1"
    echo "🎯 Using specified environment: $ENV_NAME"
elif [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
    ENV_NAME="$CONDA_DEFAULT_ENV"
    echo "🔍 Auto-detected current environment: $ENV_NAME"
else
    ENV_NAME='nek'
    echo "⚠️  No environment specified, using default: $ENV_NAME"
fi

#--- Create env -----
conda create --name "$ENV_NAME" python=3.8 -y


# Auto-detect environment path
if [[ "$CONDA_PREFIX" != "" ]]; then
    # If we're already in a conda environment, use it
    ENV_PATH="$CONDA_PREFIX"
    echo "🔍 Auto-detected current conda environment: $ENV_PATH"
    # Try to find conda installation
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
    if [[ "$CONDA_BASE" != "" ]]; then
        ENV_PATH="${CONDA_BASE}/envs/${ENV_NAME}"
        echo "🔍 Auto-detected conda base: $CONDA_BASE"
    else
        # Fallback to common paths
        ENV_PATH="/home/yuninw/apps/mpi_drl/miniforge3/envs/${ENV_NAME}"
        echo "⚠️  Using fallback path: $ENV_PATH"
    fi
else
    # Fallback to common paths
    ENV_PATH="/home/yuninw/apps/mpi_drl/miniforge3/envs/${ENV_NAME}"
    echo "⚠️  Conda not found, using fallback path: $ENV_PATH"
fi

# Verify the environment exists
if [[ ! -d "$ENV_PATH" ]]; then
    echo "❌ Environment not found at: $ENV_PATH"
    echo "💡 Please create the environment first or check the path"
    echo "💡 You can also specify the environment name: ./install_auto_env.sh your_env_name"
    exit 1
fi

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

create_bashrc_file() {
    print_status "Creating bashrc file..."
    
    # Create bashrc file
    cat > "$TARGET_BASH" << EOF
# NEK-MARL Environment Configuration
# Generated on $(date)
# Do not edit manually - this file is auto-generated

# Miniforge installation path
export NEK_MARL_ROOT="$MINIFORGE_ABS"

# Source conda
source "$MINIFORGE_ABS/bin/activate"

# Initialize conda (if not already done)
conda info >/dev/null 2>&1 || conda init bash

# Activate DRL environment by default
conda activate $ENV_NAME

# Print environment info
echo "Miniforge environment loaded:"
echo "  Miniforge: $MINIFORGE_ABS"
echo "  Active environment: $ENV_NAME"
echo "  Python version: \$(python --version 2>&1)"
EOF
    
    print_success "Environment file created: $TARGET_BASH"
}



#---- Installation ----#
echo "🚀 Installing NEK-MARL environment following DRL pattern..."
echo "📦 Environment: ${ENV_NAME}"
echo "📁 Path: ${ENV_PATH}"

### 
${ENV_PATH}/bin/pip install pip==22.3.1
echo "==================="
echo "[PIP] PIP UPDATE"
echo "==================="

${ENV_PATH}/bin/pip install setuptools==59.5.0
echo "==================="
echo "[PIP] SetupTool UPDATE"
echo "==================="

${ENV_PATH}/bin/pip install mpi4py==3.1.4 --no-cache-dir
echo "==================="
echo "[PIP] MPI4PY"
echo "==================="

${ENV_PATH}/bin/pip install stable-baselines3==1.7.0 --no-cache-dir
echo "==================="
echo "[PIP] STB3 GET"
echo "==================="

${ENV_PATH}/bin/pip install pettingzoo==1.19.0
echo "==================="
echo "[PIP] PETTING ZOO"
echo "==================="

${ENV_PATH}/bin/pip install supersuit==3.5.0
echo "==================="
echo "[PIP] SUPERSUIT GET"
echo "==================="

${ENV_PATH}/bin/pip install gym==0.24.1
echo "==================="
echo "[PIP] GYM GET"
echo "==================="

${ENV_PATH}/bin/pip install omegaconf==2.1.0
echo "==================="
echo "[PIP] OmegaConf"
echo "==================="

${ENV_PATH}/bin/pip install scipy
echo "==================="
echo "[PIP] SCIPY"
echo "==================="

${ENV_PATH}/bin/pip install matplotlib
echo "==================="
echo "[PIP] matplotlib"
echo "==================="

## Critical for buffer I/O
${ENV_PATH}/bin/pip install numpy==1.22.4
echo "==================="
echo "[PIP] NUMPY DOWNGRADE"
echo "==================="

## Tensorboard 
${ENV_PATH}/bin/pip install tensorboard
echo "==================="
echo "[PIP] TENSORBOARD"
echo "==================="

## Additional dependencies for NEK-MARL
${ENV_PATH}/bin/pip install pandas
echo "==================="
echo "[PIP] PANDAS"
echo "==================="

${ENV_PATH}/bin/pip install pyyaml
echo "==================="
echo "[PIP] PYYAML"
echo "==================="

## Install NEK-MARL package
${ENV_PATH}/bin/pip install -e . --no-deps
echo "==================="
echo "[PIP] NEK-MARL PACKAGE"
echo "==================="

## Apply Supersuit patch
python utils/patch_supersuit.py
echo "==================="
echo "[PATCH] SUPERSUIT PATCH APPLIED"
echo "==================="

### Write the environment file as bashrc file
create_bashrc_file
echo "==================="
echo "[IO] BASHRC CREATED"
echo "==================="

echo "=============================="
echo "[IO] ENVIRONMENT COMPLETE"
echo "=============================="
echo "✅ Installation completed successfully!"
echo "📋 Working version combination installed:"
echo "   - SB3: 1.7.0"
echo "   - Gym: 0.24.1" 
echo "   - PettingZoo: 1.19.0"
echo "   - Supersuit: 3.5.0"
echo "   - NumPy: 1.22.4"
echo "   - Supersuit patched for single environment optimization"
echo "🚀 Ready to use!"
