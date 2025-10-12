#!/bin/bash

# Install system dependencies
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    echo "Detected Ubuntu/Debian system"
    sudo apt-get update
    sudo apt-get install -y cmake ninja-build build-essential
    if [ "$1" = "cuda" ]; then
        sudo apt-get install -y nvidia-cuda-toolkit
    fi
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    echo "Detected CentOS/RHEL system"
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y cmake ninja-build
    if [ "$1" = "cuda" ]; then
        sudo yum install -y cuda-toolkit
    fi
elif command -v pacman &> /dev/null; then
    # Arch Linux/Manjaro
    echo "Detected Arch Linux/Manjaro system"
    sudo pacman -Sy --noconfirm cmake ninja base-devel
    if [ "$1" = "cuda" ]; then
        sudo pacman -Sy --noconfirm cuda
    fi
else
    echo "Unsupported package manager. Please install dependencies manually:"
    echo "  - cmake"
    echo "  - ninja"
    echo "  - build-essential (Ubuntu) or base-devel (Arch) or Development Tools (CentOS)"
    if [ "$1" = "cuda" ]; then
        echo "  - nvidia-cuda-toolkit (Ubuntu) or cuda (Arch) or cuda-toolkit (CentOS)"
    fi
    exit 1
fi

# Check if CUDA installation was requested and verify
if [ "$1" = "cuda" ]; then
    echo "Verifying CUDA installation..."
    if command -v nvcc &> /dev/null; then
        echo "CUDA compiler (nvcc) found: $(nvcc --version | head -n 4)"
    else
        echo "Warning: nvcc not found in PATH. CUDA may not be properly installed."
        echo "You may need to add CUDA to your PATH:"
        echo "  export PATH=/usr/local/cuda/bin:\$PATH"
        echo "  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
    fi
fi

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch numpy scikit-build-core

# Additional dependencies that might be needed
pip install pybind11

echo ""
echo "Dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Make sure your virtual environment is activated: source venv/bin/activate"
echo "2. Build the project: pip install -e ."
echo "3. Run tests: python tests/test_dummy.py"
echo ""
echo "If you plan to use CUDA, make sure it's properly set up:"
echo "  - nvcc should be in your PATH"
echo "  - CUDA libraries should be in LD_LIBRARY_PATH"
