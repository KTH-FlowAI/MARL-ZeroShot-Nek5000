#!/bin/bash

# OpenMPI + UCX Installation Script
# This script installs UCX and OpenMPI with UCX support

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
BASE_PATH=$(pwd)
MAIN_PATH="./ucx_mpi"
UCX_INSTALL="./ucx_local"
OPENMPI_VERSION="4.1.4"
OPENMPI_INSTALL="./openmpi-${OPENMPI_VERSION}-ucx"
TARGET_BASH=~/.bashrc.openmpi_ucx

# Function to check if we're in the right directory
check_directory() {
    if [ ! -f "./install_mpi_ucx.sh" ]; then
        print_error "This script must be run from the 00_Env_SetUp directory"
        exit 1
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating installation directories..."
    
    mkdir -p "$MAIN_PATH"
    cd "$MAIN_PATH"
    
    print_success "Created directory: $MAIN_PATH"
}

# Function to install UCX
install_ucx() {
    print_status "Installing UCX..."
    
    if [ -d "ucx" ]; then
        print_warning "UCX directory already exists. Removing it..."
        rm -rf ucx
    fi
    
    echo "==============="
    echo " Cloning UCX    "
    echo "==============="
    
    git clone https://github.com/openucx/ucx.git
    cd ucx
    
    # Create UCX install directory
    mkdir -p "$UCX_INSTALL"
    UCX_INSTALL_ABS=$(realpath "$UCX_INSTALL")
    print_status "UCX will be installed to: $UCX_INSTALL_ABS"
    
    # Configure and build UCX
    print_status "Configuring UCX..."
    ./autogen.sh
    ./contrib/configure-release --prefix="$UCX_INSTALL_ABS" --enable-mt
    
    print_status "Building UCX (this may take a while)..."
    make -j
    
    print_status "Installing UCX..."
    make install
    
    print_success "UCX installation completed successfully"
    
    # Go back to main directory
    cd ..
}

# Function to install OpenMPI
install_openmpi() {
    print_status "Installing OpenMPI with UCX support..."
    
    # Download OpenMPI if not already present
    if [ ! -f "openmpi-${OPENMPI_VERSION}.tar.gz" ]; then
        print_status "Downloading OpenMPI ${OPENMPI_VERSION}..."
        wget "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz"
    fi
    
    # Extract OpenMPI
    print_status "Extracting OpenMPI..."
    tar -xzvf "openmpi-${OPENMPI_VERSION}.tar.gz"
    cd "openmpi-${OPENMPI_VERSION}"
    
    # Get absolute paths
    UCX_PATH=$(realpath "../ucx/ucx_local/")
    OPENMPI_INSTALL_ABS=$(realpath "../${OPENMPI_INSTALL}")
    
    print_status "UCX PATH: $UCX_PATH"
    print_status "OpenMPI will be installed to: $OPENMPI_INSTALL_ABS"
    
    # Configure OpenMPI with UCX support
    print_status "Configuring OpenMPI with UCX support..."
    ./configure --prefix="$OPENMPI_INSTALL_ABS" \
               --with-ucx="$UCX_PATH" \
	       --without-gpfs
               CC=gcc CXX=g++ FC=gfortran
    
    # Build OpenMPI
    print_status "Building OpenMPI (this may take a while)..."
    make -j
    
    # Install OpenMPI
    print_status "Installing OpenMPI..."
    make install
    
    print_success "OpenMPI installation completed successfully"
    
    # Go back to main directory
    cd "$BASE_PATH"
}

# Function to create environment file
create_environment_file() {
    print_status "Creating environment configuration file..."
    
    # Get absolute paths
    MAIN_PATH_ABS=$(realpath "$MAIN_PATH")
    UCX_PATH_ABS=$(realpath "$MAIN_PATH/ucx/ucx_local")
    OPENMPI_PATH_ABS=$(realpath "$MAIN_PATH/${OPENMPI_INSTALL}")
    
    # Create environment file
    cat > "$TARGET_BASH" << EOF
# OpenMPI + UCX Environment Configuration
# Generated on $(date)
# Do not edit manually - this file is auto-generated

# Main installation paths
export MPI_UCX_ROOT="$MAIN_PATH_ABS"
export UCX_ROOT="$UCX_PATH_ABS"
export OPENMPI_ROOT="$OPENMPI_PATH_ABS"

# Add OpenMPI binaries to PATH
export PATH="$OPENMPI_PATH_ABS/bin:\$PATH"

# Add UCX binaries to PATH
export PATH="$UCX_PATH_ABS/bin:\$PATH"

# Add libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$OPENMPI_PATH_ABS/lib:$UCX_PATH_ABS/lib:\$LD_LIBRARY_PATH"

# Set OpenMPI prefix
export OPAL_PREFIX="$OPENMPI_PATH_ABS"

# UCX specific environment variables
export UCX_NET_DEVICES=all
export UCX_TLS=rc,sm,ud,self

# Optional: Set number of OpenMP threads
export OMP_NUM_THREADS=1

# Print environment info
echo "OpenMPI + UCX environment loaded:"
echo "  OpenMPI: $OPENMPI_PATH_ABS"
echo "  UCX: $UCX_PATH_ABS"
EOF
    
    print_success "Environment file created: $TARGET_BASH"
}

# Function to test installation
test_installation() {
    print_status "Testing installation..."
    
    # Source the environment
    source "$TARGET_BASH"
    
    # Test OpenMPI
    if command -v mpirun >/dev/null 2>&1; then
        print_success "OpenMPI found: $(mpirun --version | head -n1)"
    else
        print_error "OpenMPI not found in PATH"
        return 1
    fi
    
    # Test UCX
    if command -v ucx_info >/dev/null 2>&1; then
        print_success "UCX found: $(ucx_info -v | head -n1)"
    else
        print_error "UCX not found in PATH"
        return 1
    fi
    
    print_success "Installation test passed!"
}

# Function to show installation summary
show_summary() {
    echo
    echo "=========================================="
    echo "        OpenMPI + UCX Installation        "
    echo "=========================================="
    echo "✓ UCX installed in: $MAIN_PATH/ucx_local"
    echo "✓ OpenMPI installed in: $MAIN_PATH/${OPENMPI_INSTALL}"
    echo "✓ Environment file: $TARGET_BASH"
    echo
    echo "=========================================="
    echo "           NEXT STEPS                     "
    echo "=========================================="
    echo "1. Source the environment:"
    echo "   source $TARGET_BASH"
    echo
    echo "2. Test the installation:"
    echo "   mpirun --version"
    echo "   ucx_info -v"
    echo
    echo "3. To make the environment permanent, add to your ~/.bashrc:"
    echo "   echo 'source $TARGET_BASH' >> ~/.bashrc"
    echo
}

# Main installation function
main() {
    echo "Starting OpenMPI + UCX installation..."
    echo
    
    # Store original directory
    local ORIGINAL_DIR=$(pwd)
    
    # Check directory
    # check_directory
    
    # Create directories
    create_directories
    
    # Install UCX
    install_ucx
    
    # Install OpenMPI
    install_openmpi
   
    # Create environment file
    create_environment_file
    
    # Test installation
    test_installation
    
    # Show summary
    show_summary
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    print_success "OpenMPI + UCX installation completed successfully!"
}

# Run main function
main "$@"

