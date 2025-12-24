#!/usr/bin/env bash
# Embodied AI Architect Development Environment Setup Script
# Run this script to set up a complete development environment
#
# Usage: ./bin/setup-dev-env.sh [--all|--minimal|--soc|--help]
#
# Options:
#   --minimal   Install only essential Python dependencies
#   --all       Install everything including EDA tools and LangGraph (default)
#   --soc       Install only SoC optimizer experiment dependencies
#   --help      Show this help message

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# OSS CAD Suite version (for Yosys, Verilator, Icarus Verilog)
OSS_CAD_SUITE_VERSION="2025-12-12"
OSS_CAD_SUITE_DATE=$(echo "$OSS_CAD_SUITE_VERSION" | tr -d '-')

# Logging functions
info() { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Check if command exists
has_cmd() { command -v "$1" &>/dev/null; }

# Show help
show_help() {
    sed -n '3,12p' "$0" | sed 's/^#//' | sed 's/^ //'
    exit 0
}

# Parse arguments
INSTALL_MODE="all"
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal) INSTALL_MODE="minimal"; shift ;;
        --all) INSTALL_MODE="all"; shift ;;
        --soc) INSTALL_MODE="soc"; shift ;;
        --help|-h) show_help ;;
        *) error "Unknown option: $1" ;;
    esac
done

echo ""
echo "================================================"
echo "  Embodied AI Architect - Dev Environment Setup"
echo "================================================"
echo ""
info "Install mode: $INSTALL_MODE"
info "Project root: $PROJECT_ROOT"
echo ""

# Check OS
if [[ ! -f /etc/debian_version ]] && [[ "$(uname)" != "Darwin" ]]; then
    warn "This script is designed for Ubuntu/Debian/macOS. Some commands may not work."
fi

# ============================================================================
# System Packages (Linux only)
# ============================================================================
if [[ -f /etc/debian_version ]]; then
    info "Installing system packages..."

    ESSENTIAL_PKGS=(
        build-essential
        git
        curl
        wget
        python3
        python3-pip
        python3-venv
    )

    sudo apt-get update
    sudo apt-get install -y "${ESSENTIAL_PKGS[@]}"
    success "System packages installed"
elif [[ "$(uname)" == "Darwin" ]]; then
    info "macOS detected - assuming Homebrew and Python are installed"
    if ! has_cmd python3; then
        error "Python 3 not found. Install with: brew install python"
    fi
fi

# ============================================================================
# Python Virtual Environment
# ============================================================================
info "Setting up Python virtual environment..."

cd "$PROJECT_ROOT"

if [[ ! -d .venv ]]; then
    python3 -m venv .venv
    success "Created virtual environment"
else
    success "Virtual environment already exists"
fi

# Activate venv
source .venv/bin/activate

info "Upgrading pip..."
pip install --upgrade pip

# ============================================================================
# Python Dependencies
# ============================================================================
info "Installing Python dependencies..."

case "$INSTALL_MODE" in
    minimal)
        pip install -e .
        success "Core dependencies installed"
        ;;
    soc)
        pip install -e ".[dev]"
        pip install langgraph langchain-anthropic
        success "SoC optimizer dependencies installed"
        ;;
    all)
        pip install -e ".[all,dev]"
        pip install langgraph langchain-anthropic
        success "All Python dependencies installed"
        ;;
esac

# ============================================================================
# OSS CAD Suite (for SoC optimizer and EDA workflows)
# ============================================================================
if [[ "$INSTALL_MODE" == "all" ]] || [[ "$INSTALL_MODE" == "soc" ]]; then
    OSS_CAD_SUITE_PATH="/opt/oss-cad-suite"

    if [[ -d "$OSS_CAD_SUITE_PATH" ]]; then
        success "OSS CAD Suite already installed at $OSS_CAD_SUITE_PATH"
    else
        info "Installing OSS CAD Suite (Yosys, Verilator, Icarus Verilog)..."

        if [[ "$(uname)" == "Darwin" ]]; then
            OSS_CAD_TARBALL="oss-cad-suite-darwin-x64-${OSS_CAD_SUITE_DATE}.tgz"
        else
            OSS_CAD_TARBALL="oss-cad-suite-linux-x64-${OSS_CAD_SUITE_DATE}.tgz"
        fi

        OSS_CAD_URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${OSS_CAD_SUITE_VERSION}/${OSS_CAD_TARBALL}"

        info "Downloading from: $OSS_CAD_URL"
        wget -q --show-progress -O "/tmp/${OSS_CAD_TARBALL}" "$OSS_CAD_URL" || \
            curl -L -o "/tmp/${OSS_CAD_TARBALL}" "$OSS_CAD_URL"

        info "Extracting to /opt (requires sudo)..."
        sudo tar -xzf "/tmp/${OSS_CAD_TARBALL}" -C /opt
        rm "/tmp/${OSS_CAD_TARBALL}"

        success "OSS CAD Suite installed"
    fi

    # Add to PATH for current session
    export PATH="$OSS_CAD_SUITE_PATH/bin:$PATH"

    # ========================================================================
    # Patch venv activate script to include OSS CAD Suite
    # ========================================================================
    info "Patching venv activation script to include OSS CAD Suite..."
    ACTIVATE_SCRIPT="$PROJECT_ROOT/.venv/bin/activate"

    if ! grep -q "oss-cad-suite" "$ACTIVATE_SCRIPT"; then
        cat >> "$ACTIVATE_SCRIPT" << 'EOF'

# Add OSS CAD Suite to PATH (for Yosys, Verilator, Icarus Verilog)
if [ -d "/opt/oss-cad-suite/bin" ]; then
    PATH="/opt/oss-cad-suite/bin:$PATH"
    export PATH
fi
EOF
        success "Patched activate script"
    else
        success "Activate script already patched"
    fi
fi

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "================================================"
echo "  Verifying Installation"
echo "================================================"
echo ""

verify_tool() {
    local name=$1
    local cmd=$2
    local version
    if has_cmd "$cmd"; then
        version=$($cmd --version 2>&1 | head -1)
        success "$name: $version"
        return 0
    else
        warn "$name: NOT FOUND"
        return 1
    fi
}

verify_python_pkg() {
    local name=$1
    if python -c "import $name" 2>/dev/null; then
        local version=$(python -c "import $name; print(getattr($name, '__version__', 'installed'))" 2>/dev/null)
        success "$name: $version"
        return 0
    else
        warn "$name: NOT INSTALLED"
        return 1
    fi
}

# Core tools
verify_tool "Python" python3
verify_tool "Pip" pip

# Python packages
verify_python_pkg "torch"
verify_python_pkg "pydantic"
verify_python_pkg "anthropic"

if [[ "$INSTALL_MODE" == "all" ]] || [[ "$INSTALL_MODE" == "soc" ]]; then
    verify_python_pkg "langgraph"

    # EDA tools
    verify_tool "Yosys" yosys
    verify_tool "Verilator" verilator
    verify_tool "Icarus Verilog" iverilog
fi

# Verify CLI entry point
if has_cmd embodied-ai; then
    success "CLI entry point: embodied-ai"
else
    warn "CLI entry point not found (run 'pip install -e .' again)"
fi

echo ""
echo "================================================"
echo "  Setup Complete!"
echo "================================================"
echo ""
info "To activate the environment in a new shell:"
echo ""
echo "    cd $PROJECT_ROOT"
echo "    source .venv/bin/activate"
echo ""
info "Quick test commands:"
echo ""
echo "    embodied-ai --help              # Show CLI help"
echo "    pytest tests/                   # Run tests"
if [[ "$INSTALL_MODE" == "all" ]] || [[ "$INSTALL_MODE" == "soc" ]]; then
echo "    cd experiments/langgraph/soc_optimizer"
echo "    python workflow.py              # Run SoC optimizer demo"
fi
echo ""
success "Happy development!"
