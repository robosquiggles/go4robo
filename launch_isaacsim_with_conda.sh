#!/bin/bash

# Default environment name
DEFAULT_ENV="isaac"

# Check if environment name was provided as parameter
if [ $# -ge 1 ]; then
    ENV_NAME="$1"
    # Shift the arguments so that $@ can be passed to isaac-sim.sh later
    shift
else
    ENV_NAME="$DEFAULT_ENV"
    echo "No environment name provided, using default: $DEFAULT_ENV"
fi

# Find conda dynamically
CONDA_PATH=$(which conda)
if [ -z "$CONDA_PATH" ]; then
    echo "Error: conda not found in PATH. Please install conda or add it to your PATH."
    exit 1
fi

# Extract conda base directory (removes '/bin/conda' from path)
CONDA_BASE=$(dirname $(dirname "$CONDA_PATH"))
echo "Using conda installation found at: $CONDA_BASE"

# Source conda initialization script
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Activate the specified conda environment
echo "Activating conda environment: $ENV_NAME"
conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate $ENV_NAME environment. Make sure it exists."
    echo "Create it with: conda create -n $ENV_NAME python=3.10"
    exit 1
fi

# Set PYTHONPATH to include your conda environment's site-packages
SITE_PACKAGES="$CONDA_BASE/envs/$ENV_NAME/lib/python3.10/site-packages"
export PYTHONPATH=$SITE_PACKAGES:$PYTHONPATH
echo "Added $SITE_PACKAGES to PYTHONPATH"

# Launch Isaac Sim with your environment
cd /home/rachael/Documents/GitHub/go4robo/isaacsim
echo "Launching Isaac Sim..."
./isaac-sim.sh "$@"