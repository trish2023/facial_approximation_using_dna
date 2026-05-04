#!/usr/bin/env bash
# ===========================================================================
#  setup_env.sh — Idempotent environment bootstrap for the
#  DNA-Based Facial Approximation pipeline.
#
#  Usage:  bash setup_env.sh
#          (safe to re-run — skips steps that are already complete)
# ===========================================================================

# Re-exec under bash if the current shell is not bash (e.g. invoked via sh)
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -e
set -u
set -o pipefail

# ── Configuration ──────────────────────────────────────────────────────────
ENV_NAME="dna_facial"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/requirements.txt"

CONDA_PACKAGES=(
    bcftools
    samtools
    htslib      # provides tabix
    plink2
)

VERIFY_TOOLS=(
    "bcftools   : bcftools --version"
    "samtools   : samtools --version"
    "tabix      : tabix --version"
    "plink2     : plink2 --version"
    "python     : python --version"
    "pip        : pip --version"
)

# ── Colour helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'   # No Colour

info()    { echo -e "${BOLD}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
ok()      { echo -e "${GREEN}[  OK]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── Locate and initialise conda ───────────────────────────────────────────
# Scripts don't source .bashrc, so conda is usually not on PATH.
# Try the common install locations and initialise the first one found.

_try_conda_init() {
    local conda_exe="$1"
    if [ -x "${conda_exe}" ]; then
        eval "$("${conda_exe}" shell.bash hook)"
        return 0
    fi
    return 1
}

if ! command -v conda &>/dev/null; then
    info "conda not on PATH — searching common install locations …"

    FOUND=false
    SEARCH_PATHS=(
        "${CONDA_EXE:-}"                                    # set by conda init in parent shell
        "${HOME}/miniconda3/bin/conda"
        "${HOME}/miniforge3/bin/conda"
        "${HOME}/anaconda3/bin/conda"
        "${HOME}/mambaforge/bin/conda"
        "/opt/conda/bin/conda"
        "/opt/miniconda3/bin/conda"
        "/opt/anaconda3/bin/conda"
        "/usr/local/miniconda3/bin/conda"
        "${HOME}/AppData/Local/miniconda3/Scripts/conda.exe"        # Windows (Git Bash / MSYS2)
        "${HOME}/AppData/Local/anaconda3/Scripts/conda.exe"
        "/c/ProgramData/miniconda3/Scripts/conda.exe"
        "/c/ProgramData/Anaconda3/Scripts/conda.exe"
        "${USERPROFILE:-}/miniconda3/Scripts/conda.exe"
        "${USERPROFILE:-}/anaconda3/Scripts/conda.exe"
    )

    for candidate in "${SEARCH_PATHS[@]}"; do
        [ -z "${candidate}" ] && continue
        if _try_conda_init "${candidate}"; then
            ok "Found conda at ${candidate}"
            FOUND=true
            break
        fi
    done

    if [ "${FOUND}" = false ]; then
        fail "conda not found. Searched these locations:"
        for p in "${SEARCH_PATHS[@]}"; do
            [ -n "${p}" ] && fail "  ${p}"
        done
        echo ""
        fail "Install Miniconda (https://docs.conda.io/en/latest/miniconda.html)"
        fail "or set CONDA_EXE to your conda binary before running this script:"
        fail "  export CONDA_EXE=/path/to/conda && bash setup_env.sh"
        exit 1
    fi
else
    eval "$(conda shell.bash hook)"
fi

info "Using conda: $(command -v conda)"

# ── Step 1: Create conda environment (skip if it already exists) ──────────
if conda env list | grep -qw "${ENV_NAME}"; then
    info "Conda environment '${ENV_NAME}' already exists — skipping creation."
else
    info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION} …"
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
    ok "Environment created."
fi

conda activate "${ENV_NAME}"
info "Activated environment: ${CONDA_DEFAULT_ENV}"

# ── Step 2: Install bioinformatics CLI tools via conda-forge ──────────────
MISSING_CONDA=()
for pkg in "${CONDA_PACKAGES[@]}"; do
    if conda list -n "${ENV_NAME}" "^${pkg}$" 2>/dev/null | grep -qw "${pkg}"; then
        info "${pkg} already installed — skipping."
    else
        MISSING_CONDA+=("${pkg}")
    fi
done

if [ ${#MISSING_CONDA[@]} -gt 0 ]; then
    info "Installing conda-forge packages: ${MISSING_CONDA[*]} …"
    conda install -y -c conda-forge "${MISSING_CONDA[@]}"
    ok "Conda packages installed."
else
    info "All conda-forge packages already present."
fi

# ── Step 3: Install Python packages from requirements.txt via pip ─────────
if [ ! -f "${REQUIREMENTS}" ]; then
    fail "requirements.txt not found at ${REQUIREMENTS}"
    exit 1
fi

info "Installing / upgrading Python packages from requirements.txt …"
pip install --upgrade pip
pip install -r "${REQUIREMENTS}"
ok "Python packages installed."

# ── Step 4 & 5: Verify every tool is on PATH ─────────────────────────────
echo ""
info "Verifying tool availability …"
echo "────────────────────────────────────────────────────"

MISSING=()
for entry in "${VERIFY_TOOLS[@]}"; do
    TOOL_NAME="${entry%%:*}"
    TOOL_NAME="$(echo "${TOOL_NAME}" | xargs)"   # trim whitespace
    TOOL_CMD="${entry#*:}"
    TOOL_CMD="$(echo "${TOOL_CMD}" | xargs)"

    # tabix prints its usage to stderr and exits non-zero when called with
    # --version, so we capture both streams and accept either exit code.
    if OUTPUT=$(${TOOL_CMD} 2>&1) || true; then
        VERSION_LINE=$(echo "${OUTPUT}" | head -n 1)
        ok "${TOOL_NAME}  →  ${VERSION_LINE}"
    fi

    # Definitive reachability check (the command binary must exist on PATH)
    BINARY="${TOOL_CMD%% *}"
    if ! command -v "${BINARY}" &>/dev/null; then
        MISSING+=("${TOOL_NAME}")
    fi
done

# Also verify key Python libraries can be imported
PYTHON_LIBS=( allel pandas numpy sklearn cv2 dlib pysam cyvcf2 )
for lib in "${PYTHON_LIBS[@]}"; do
    if python -c "import ${lib}" &>/dev/null; then
        ok "python:${lib}  →  importable"
    else
        fail "python:${lib}  →  import failed"
        MISSING+=("python:${lib}")
    fi
done

echo "────────────────────────────────────────────────────"

# ── Final report ──────────────────────────────────────────────────────────
if [ ${#MISSING[@]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}${BOLD}✔  All tools and libraries verified successfully.${NC}"
    echo -e "${GREEN}   Activate the environment with:  conda activate ${ENV_NAME}${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}${BOLD}✘  The following tools/libraries are missing or broken:${NC}"
    for m in "${MISSING[@]}"; do
        echo -e "${RED}     • ${m}${NC}"
    done
    echo ""
    exit 1
fi
