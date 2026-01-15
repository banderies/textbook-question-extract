#!/bin/bash
# =============================================================================
# Anki Q&A Extractor - Environment Setup
# =============================================================================
#
# This script creates and configures the conda environment for the Anki
# Q&A extraction pipeline.
#
# Usage:
#   ./setup_env.sh           # Create environment and install dependencies
#   ./setup_env.sh --check   # Check if environment is ready
#
# After setup, activate the environment with:
#   conda activate anki-extractor
#
# =============================================================================

set -e  # Exit on error

# Configuration
ENV_NAME="anki-extractor"
PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find conda installation
find_conda() {
    # Check common conda locations
    local conda_paths=(
        "$HOME/opt/miniconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    )

    for path in "${conda_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    # Try to find via which
    if command -v conda &> /dev/null; then
        local conda_base=$(conda info --base 2>/dev/null)
        if [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            echo "$conda_base/etc/profile.d/conda.sh"
            return 0
        fi
    fi

    return 1
}

# Initialize conda
init_conda() {
    local conda_sh=$(find_conda)
    if [ -z "$conda_sh" ]; then
        echo -e "${RED}Error: Could not find conda installation${NC}"
        echo "Please install Miniconda or Anaconda first:"
        echo "  https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi

    echo -e "${GREEN}Found conda at: $conda_sh${NC}"
    source "$conda_sh"
}

# Check if environment exists
env_exists() {
    conda env list | grep -q "^$ENV_NAME "
}

# Check if all dependencies are installed
check_dependencies() {
    echo "Checking dependencies..."

    local missing=0

    # Check Python packages
    python -c "import docling" 2>/dev/null || { echo -e "  ${RED}✗ docling${NC}"; missing=1; }
    python -c "import fitz" 2>/dev/null || { echo -e "  ${RED}✗ pymupdf${NC}"; missing=1; }
    python -c "import anthropic" 2>/dev/null || { echo -e "  ${RED}✗ anthropic${NC}"; missing=1; }
    python -c "import genanki" 2>/dev/null || { echo -e "  ${RED}✗ genanki${NC}"; missing=1; }

    if [ $missing -eq 0 ]; then
        echo -e "  ${GREEN}✓ All dependencies installed${NC}"
        return 0
    else
        return 1
    fi
}

# Create environment
create_environment() {
    echo -e "${YELLOW}Creating conda environment: $ENV_NAME${NC}"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
}

# Install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    conda activate "$ENV_NAME"
    pip install -r "$SCRIPT_DIR/requirements.txt"
}

# Main check mode
check_mode() {
    init_conda

    echo "=== Environment Check ==="

    if env_exists; then
        echo -e "${GREEN}✓ Environment '$ENV_NAME' exists${NC}"
        conda activate "$ENV_NAME"
        check_dependencies

        # Check for API key
        if [ -n "$ANTHROPIC_API_KEY" ]; then
            echo -e "  ${GREEN}✓ ANTHROPIC_API_KEY is set${NC}"
        else
            echo -e "  ${YELLOW}⚠ ANTHROPIC_API_KEY not set${NC}"
            echo "    Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        fi
    else
        echo -e "${RED}✗ Environment '$ENV_NAME' does not exist${NC}"
        echo "  Run: ./setup_env.sh"
        exit 1
    fi
}

# Main setup mode
setup_mode() {
    init_conda

    echo "=== Anki Q&A Extractor Setup ==="
    echo ""

    if env_exists; then
        echo -e "${YELLOW}Environment '$ENV_NAME' already exists${NC}"
        read -p "Reinstall dependencies? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_dependencies
        fi
    else
        create_environment
        install_dependencies
    fi

    echo ""
    echo -e "${GREEN}=== Setup Complete ===${NC}"
    echo ""
    echo "To use the environment:"
    echo "  conda activate $ENV_NAME"
    echo ""
    echo "Set your API key:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    echo ""
    echo "Run the extraction pipeline:"
    echo "  streamlit run src/review_gui.py"
}

# Parse arguments
case "${1:-}" in
    --check)
        check_mode
        ;;
    --help|-h)
        echo "Usage: $0 [--check] [--help]"
        echo ""
        echo "Options:"
        echo "  --check    Check if environment is ready"
        echo "  --help     Show this help message"
        echo ""
        echo "Without arguments, creates the environment and installs dependencies."
        ;;
    *)
        setup_mode
        ;;
esac
