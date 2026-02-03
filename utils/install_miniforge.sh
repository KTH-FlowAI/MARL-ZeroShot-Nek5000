#!/bin/bash

# Python/miniforge Installation Script
# This script installs miniforge and creates a Python environment for DRL

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Configuration
MINIFORGE_VERSION="latest"
PYTHON_VERSION="3.8"
ENV_NAME="nek"
MINIFORGE_INSTALL="./miniforge3"
TARGET_BASH=~/.bashrc.miniforge

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "./install_python.sh" ]; then
        print_error "This script must be run from the 00_Env_SetUp directory"
        exit 1
    fi
}

# Function to check system architecture
check_architecture() {
    print_status "Checking system architecture..."
    
    local arch=$(uname -m)
    case $arch in
        x86_64)
            print_success "Detected x86_64 architecture"
            MINIFORGE_ARCH="Linux-x86_64"
            ;;
        aarch64|arm64)
            print_success "Detected ARM64 architecture"
            MINIFORGE_ARCH="Linux-aarch64"
            ;;
        *)
            print_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
}

# Function to check if miniforge is already installed
check_existing_installation() {
    if [ -d "$MINIFORGE_INSTALL" ]; then
        print_warning "Miniforge installation found at $MINIFORGE_INSTALL"
        read -p "Do you want to remove the existing installation and reinstall? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing miniforge installation..."
            rm -rf "$MINIFORGE_INSTALL"
            print_success "Existing installation removed"
        else
            print_status "Using existing installation"
            return 0
        fi
    fi
}

# Function to download miniforge
download_miniforge() {
    print_status "Downloading miniforge..."
    
    local download_url="https://github.com/conda-forge/miniforge/releases/${MINIFORGE_VERSION}/download/Miniforge3-${MINIFORGE_ARCH}.sh"
    
    print_status "Download URL: $download_url"
    
    if ! wget -O miniforge_installer.sh "$download_url"; then
        print_error "Failed to download miniforge"
        exit 1
    fi
    
    chmod +x miniforge_installer.sh
    print_success "Miniforge downloaded successfully"
}

# Function to install miniforge
install_miniforge() {
    print_status "Installing miniforge..."
    
    # Run the installer
    if ! bash miniforge_installer.sh -b -p "$MINIFORGE_INSTALL"; then
        print_error "Failed to install miniforge"
        exit 1
    fi
    
    # Clean up installer
    rm miniforge_installer.sh
    
    print_success "Miniforge installed successfully to $MINIFORGE_INSTALL"
}

# Function to initialize conda
initialize_conda() {
    print_status "Initializing conda..."
    
    # Source conda
    source "$MINIFORGE_INSTALL/etc/profile.d/conda.sh"
    
    # Initialize conda for bash
    conda init bash
    
    print_success "Conda initialized successfully"
}

# Function to create Python environment
create_python_environment() {
    print_status "Creating Python environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    
    # Source conda
    source "$MINIFORGE_INSTALL/etc/profile.d/conda.sh"
    
    # Create environment
    if conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y; then
        print_success "Python environment '$ENV_NAME' created successfully"
    else
        print_error "Failed to create Python environment"
        exit 1
    fi
    
    # Install basic packages
    print_status "Installing basic packages..."
    conda activate "$ENV_NAME"
    
    # Install common scientific computing packages
    conda install -y numpy scipy matplotlib pandas jupyter
    conda install -y pip
    
    print_success "Basic packages installed successfully"
}

# Function to install DRL packages using the existing script
install_drl_packages() {
    print_status "Installing DRL-specific packages using 03-install-python-env.sh..."
    
    if [ ! -f "./03-install-python-env.sh" ]; then
        print_warning "03-install-python-env.sh not found. Installing basic packages only."
        return 0
    fi
    
    # Check if the script needs to be updated for the new miniforge path
    local script_content=$(cat "./03-install-python-env.sh")
    
    # Update the script to use the correct miniforge path
    local updated_script="${script_content//miniconda3/${MINIFORGE_INSTALL}}"
    
    # Create a temporary updated script
    local temp_script="/tmp/install_drl_packages_temp.sh"
    echo "$updated_script" > "$temp_script"
    chmod +x "$temp_script"
    
    # Source conda and activate environment
    source "$MINIFORGE_INSTALL/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    
    # Run the updated script
    print_status "Running DRL package installation..."
    if bash "$temp_script"; then
        print_success "DRL packages installed successfully"
    else
        print_error "Failed to install DRL packages"
        rm -f "$temp_script"
        return 1
    fi
    
    # Clean up temporary script
    rm -f "$temp_script"
}

# Function to create environment file
create_environment_file() {
    print_status "Creating environment configuration file..."
    
    # Get absolute path
    MINIFORGE_ABS=$(realpath "$MINIFORGE_INSTALL")
    
    # Create environment file
    cat > "$TARGET_BASH" << EOF
# Miniforge Environment Configuration
# Generated on $(date)
# Do not edit manually - this file is auto-generated

# Miniforge installation path
export MINIFORGE_ROOT="$MINIFORGE_ABS"

# Source conda
source "$MINIFORGE_ABS/etc/profile.d/conda.sh"

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

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    # Source the environment
    source "$TARGET_BASH"
    
    # Test conda
    if command -v conda >/dev/null 2>&1; then
        print_success "Conda found: $(conda --version)"
    else
        print_error "Conda not found in PATH"
        return 1
    fi
    
    # Test Python
    if command -v python >/dev/null 2>&1; then
        print_success "Python found: $(python --version)"
    else
        print_error "Python not found in PATH"
        return 1
    fi
    
    # Test environment activation
    if conda info --envs | grep -q "^\* $ENV_NAME"; then
        print_success "Environment '$ENV_NAME' is active"
    else
        print_error "Environment '$ENV_NAME' is not active"
        return 1
    fi
    
    # Test key DRL packages
    print_status "Testing key DRL packages..."
    python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>/dev/null || print_warning "NumPy not found"
    python -c "import scipy; print('✓ SciPy:', scipy.__version__)" 2>/dev/null || print_warning "SciPy not found"
    python -c "import stable_baselines3; print('✓ Stable-Baselines3:', stable_baselines3.__version__)" 2>/dev/null || print_warning "Stable-Baselines3 not found"
    
    print_success "Installation test passed!"
}

# Function to show installation summary
show_summary() {
    echo
    echo "=========================================="
    echo "        Python/miniforge Installation     "
    echo "=========================================="
    echo "✓ Miniforge installed in: $MINIFORGE_INSTALL"
    echo "✓ Python environment created: $ENV_NAME"
    echo "✓ Environment file: $TARGET_BASH"
    echo "✓ DRL packages installed using 03-install-python-env.sh"
    echo
    echo "=========================================="
    echo "           NEXT STEPS                     "
    echo "=========================================="
    echo "1. Source the environment:"
    echo "   source $TARGET_BASH"
    echo
    echo "2. Test the installation:"
    echo "   conda --version"
    echo "   python --version"
    echo "   conda info --envs"
    echo
    echo "3. To make the environment permanent, add to your ~/.bashrc:"
    echo "   echo 'source $TARGET_BASH' >> ~/.bashrc"
    echo
    echo "4. Test DRL packages:"
    echo "   python -c 'import stable_baselines3; print(stable_baselines3.__version__)'"
    echo "   python -c 'import pettingzoo; print(pettingzoo.__version__)'"
    echo
}

# Function to ask about additional packages
ask_additional_packages() {
    print_status "Would you like to install additional packages for DRL?"
    read -p "Install common DRL packages (PyTorch, TensorFlow, Gym)? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing additional DRL packages..."
        
        # Source conda and activate environment
        source "$MINIFORGE_INSTALL/etc/profile.d/conda.sh"
        conda activate "$ENV_NAME"
        
        # Install PyTorch
        print_status "Installing PyTorch..."
        conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
        
        # Install TensorFlow
        print_status "Installing TensorFlow..."
        conda install -y tensorflow
        
        # Install Gym and other DRL packages
        print_status "Installing Gym and other DRL packages..."
        pip install gymnasium stable-baselines3
        
        print_success "Additional packages installed successfully"
    fi
}

# Main installation function
main() {
    echo "Starting Python/miniforge installation..."
    echo
    
    # Check directory
#    check_directory
    
    # Check architecture
    check_architecture
    
    # Check existing installation
    check_existing_installation
    
    # Download miniforge
    download_miniforge
    
    # Install miniforge
    install_miniforge
    
    # Initialize conda
    initialize_conda
    
    # Create environment file
    # create_environment_file
    
    # Test installation
    # test_installation
    
    # Show summary
    # show_summary
    
    print_success "Python/miniforge installation completed successfully!"
}

# Run main function
main "$@"
