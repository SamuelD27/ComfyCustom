#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors & Symbols ─────────────────────────────────────────
RST='\033[0m'
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
BLUE='\033[34m'
MAGENTA='\033[35m'
CHECK="${GREEN}\u2714${RST}"
WARN="${YELLOW}\u26A0${RST}"
CROSS="${RED}\u2718${RST}"
ARROW="${CYAN}\u25B6${RST}"
HLINE="${DIM}$(printf '\u2500%.0s' {1..52})${RST}"

ok()   { echo -e "  ${CHECK}  $1"; }
warn() { echo -e "  ${WARN}  ${YELLOW}$1${RST}"; }
fail() { echo -e "  ${CROSS}  ${RED}$1${RST}"; }
info() { echo -e "  ${ARROW}  $1"; }

# Animated progress bar for a task
# Usage: run_with_bar "Label" command args...
run_with_bar() {
    local label="$1"; shift
    local frames=("${CYAN}  \u2588\u2588${DIM}\u2591\u2591\u2591\u2591\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2588\u2588${DIM}\u2591\u2591\u2591\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2588\u2588${DIM}\u2591\u2591\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2591\u2588\u2588${DIM}\u2591\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2591\u2591\u2588\u2588${DIM}\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2591\u2591\u2591\u2588\u2588${DIM}\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2591\u2591\u2591\u2591\u2588\u2588${RST}"
                   "${CYAN}  \u2591\u2591\u2591\u2591\u2591\u2588\u2588${DIM}\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2591\u2591\u2588\u2588${DIM}\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2591\u2588\u2588${DIM}\u2591\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2591\u2588\u2588${DIM}\u2591\u2591\u2591\u2591${RST}"
                   "${CYAN}  \u2591\u2588\u2588${DIM}\u2591\u2591\u2591\u2591\u2591${RST}")
    local i=0

    "$@" &>/dev/null &
    local pid=$!

    while kill -0 "$pid" 2>/dev/null; do
        echo -ne "\r  ${frames[$((i % ${#frames[@]}))]}  ${DIM}${label}${RST}  "
        i=$((i + 1))
        sleep 0.1
    done

    wait "$pid" 2>/dev/null
    local rc=$?
    echo -ne "\r\033[2K"
    return $rc
}

# ── Banner ────────────────────────────────────────────────────
clear
echo ""
echo -e "${CYAN}${BOLD}"
cat << 'BANNER'
     ______                 ____       __  ______
    / ____/___  ____ ___   / __/_  __ / / / /  _/
   / /   / __ \/ __ `__ \ / /_ / / / / / / // /
  / /___/ /_/ / / / / / // __// /_/ / /_/ // /
  \____/\____/_/ /_/ /_//_/   \__, /\____/___/
                              /____/
BANNER
echo -e "${RST}"

VERSION=$(grep '^version' "$SCRIPT_DIR/pyproject.toml" 2>/dev/null | head -1 | sed 's/.*"\(.*\)".*/\1/')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

# Detect VRAM -- discrete GPUs report via nvidia-smi, unified memory (GB10) via system RAM
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ -z "$GPU_MEM" || "$GPU_MEM" == *"N/A"* || "$GPU_MEM" == *"Not"* ]]; then
    # Unified memory architecture -- total system RAM is shared with GPU
    GPU_MEM_GB=$(free -g | awk '/Mem:/{print $2}')
    MEM_LABEL="unified"
else
    GPU_MEM_GB=$((GPU_MEM / 1024))
    MEM_LABEL="VRAM"
fi

echo -e "  ${DIM}v${VERSION:-?}${RST}  ${DIM}|${RST}  ${BOLD}${GPU_NAME}${RST}  ${DIM}|${RST}  ${GREEN}${GPU_MEM_GB} GB ${MEM_LABEL}${RST}"
echo -e "  ${HLINE}"
echo ""

# ── Activate venv ─────────────────────────────────────────────
source .venv/bin/activate

# ── Bootstrap dependencies ───────────────────────────────────
# aria2c -- parallel downloader for model manager
if ! command -v aria2c &>/dev/null; then
    if run_with_bar "Installing aria2" sudo apt-get install -y -qq aria2; then
        ok "aria2 installed"
    else
        warn "aria2 install failed (model downloads will be slower)"
    fi
fi

# gum -- pretty TUI prompts for model manager
if ! command -v gum &>/dev/null; then
    _GUM_VER="0.17.0"
    _GUM_URL="https://github.com/charmbracelet/gum/releases/download/v${_GUM_VER}/gum_${_GUM_VER}_Linux_arm64.tar.gz"
    _GUM_TMP=$(mktemp -d)
    if run_with_bar "Installing gum" bash -c "curl -fsSL '${_GUM_URL}' | tar xz -C '${_GUM_TMP}' && sudo install -m 755 '${_GUM_TMP}/gum' /usr/local/bin/gum"; then
        ok "gum installed"
    else
        warn "gum install failed (model manager will use plain prompts)"
    fi
    rm -rf "${_GUM_TMP}"
fi

# rich -- colorized tables for model manager
if ! python -c "import rich" &>/dev/null; then
    if run_with_bar "Installing rich" uv pip install rich; then
        ok "rich installed"
    else
        warn "rich install failed (model manager will use plain output)"
    fi
fi

# ── Log file & session ────────────────────────────────────────
LOG_FILE="$SCRIPT_DIR/comfyui.log"
SESSION_FILE="/tmp/comfyui_session_$$"
export __COMFY_CLI_SESSION__="$SESSION_FILE"

cleanup() {
    local manifest="$SCRIPT_DIR/models/.tmp_models_manifest.json"
    if [[ -f "$manifest" ]]; then
        python "$SCRIPT_DIR/scripts/model_manager.py" --cleanup 2>/dev/null || true
    fi
    rm -f "$SESSION_FILE" "${SESSION_FILE}.reboot"
    echo ""
    echo -e "  ${DIM}ComfyUI stopped. Temporary models cleaned up.${RST}"
    echo ""
}
trap cleanup EXIT

> "$LOG_FILE"

# ── Pre-flight checks ────────────────────────────────────────
echo -e "  ${BOLD}Pre-flight checks${RST}"
echo ""

# 1. Manager cache
MANAGER_CACHE="$SCRIPT_DIR/user/__manager/cache"
if [[ -d "$MANAGER_CACHE" ]]; then
    rm -rf "${MANAGER_CACHE:?}"/*
    ok "Manager cache cleared"
else
    ok "Manager cache clean"
fi

# 2. Frontend version sync
COMFY_SETTINGS="$SCRIPT_DIR/user/default/comfy.settings.json"
if [[ -f "$COMFY_SETTINGS" ]]; then
    ACTUAL_VER=$(python -c "from importlib.metadata import version; print(version('comfyui-frontend-package'))" 2>/dev/null || true)
    if [[ -n "$ACTUAL_VER" ]]; then
        STORED_VER=$(python -c "import json; print(json.load(open('$COMFY_SETTINGS')).get('Comfy.InstalledVersion',''))" 2>/dev/null || true)
        if [[ "$STORED_VER" != "$ACTUAL_VER" ]]; then
            python -c "
import json
with open('$COMFY_SETTINGS') as f: d=json.load(f)
d['Comfy.InstalledVersion']='$ACTUAL_VER'
with open('$COMFY_SETTINGS','w') as f: json.dump(d,f,indent=4)
"
            ok "Frontend synced  ${DIM}${STORED_VER:-?} -> ${ACTUAL_VER}${RST}"
        else
            ok "Frontend up to date  ${DIM}v${ACTUAL_VER}${RST}"
        fi
    fi
fi

# 3. Models symlink
if [[ -d "$SCRIPT_DIR/models" && ! -L "$SCRIPT_DIR/models" ]]; then
    rm -rf "$SCRIPT_DIR/models"
    ln -s "$HOME/models/comfyui" "$SCRIPT_DIR/models"
    warn "Restored models/ symlink"
else
    ok "Models directory OK"
fi

# 4. GPU check
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
if [[ "$MEM_LABEL" == "unified" ]]; then
    # Unified memory -- show system RAM usage
    MEM_USED_GB=$(free -g | awk '/Mem:/{print $3}')
    if (( MEM_USED_GB > GPU_MEM_GB / 2 )); then
        warn "Memory: ${MEM_USED_GB}/${GPU_MEM_GB} GB in use"
    else
        ok "GPU ready  ${DIM}${MEM_USED_GB}/${GPU_MEM_GB} GB used${RST}"
    fi
else
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    GPU_MEM_USED_GB=$((GPU_MEM_USED / 1024))
    if (( GPU_MEM_USED > GPU_MEM / 2 )); then
        warn "VRAM: ${GPU_MEM_USED_GB}/${GPU_MEM_GB} GB in use"
    else
        ok "GPU ready  ${DIM}${GPU_MEM_USED_GB}/${GPU_MEM_GB} GB used${RST}"
    fi
fi

# 5. Custom nodes count
NODE_COUNT=$(find "$SCRIPT_DIR/custom_nodes" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
ok "${NODE_COUNT} custom nodes loaded"

echo ""
echo -e "  ${HLINE}"
echo ""

# ── Model Manager ────────────────────────────────────────────
python scripts/model_manager.py || {
    rc=$?
    if [[ $rc -eq 130 ]]; then
        echo -e "  ${YELLOW}Model selection cancelled.${RST}"
    else
        warn "Model manager exited with code $rc"
    fi
}

echo -e "  ${HLINE}"
echo ""

# ── Launch info ───────────────────────────────────────────────
PORT="8188"
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--port" ]]; then
        PORT="$arg"
        break
    fi
    prev_arg="$arg"
done

echo -e "  ${BOLD}${GREEN}Starting ComfyUI${RST}"
echo ""
echo -e "  ${BLUE}URL${RST}     http://0.0.0.0:${PORT}"
echo -e "  ${BLUE}Log${RST}     ${DIM}tail -f ${LOG_FILE}${RST}"
echo ""
echo -e "  ${HLINE}"
echo ""

# ── Main loop ─────────────────────────────────────────────────
while true; do
    rm -f "${SESSION_FILE}.reboot"

    python main.py \
        --listen \
        --port 8188 \
        --enable-cors-header \
        --highvram \
        --log-stdout \
        "$@" 2>&1 | tee -a "$LOG_FILE" | while IFS= read -r line; do
        case "$line" in
            *execution_cached*)
                # suppress noisy cache lines
                ;;
            *"Prompt executed"*)
                echo -e "  ${GREEN}>>>${RST}  ${GREEN}${line}${RST}"
                ;;
            *ERROR*|*error*|*Error*)
                echo -e "  ${RED}!!!${RST}  ${RED}${line}${RST}"
                ;;
            *WARNING*|*warning*)
                echo -e "  ${YELLOW}---${RST}  ${YELLOW}${line}${RST}"
                ;;
            *"Starting server"*|*"To see the GUI"*)
                echo -e "  ${GREEN}${BOLD}${line}${RST}"
                ;;
            *"got prompt"*)
                echo -e "  ${CYAN}>>>${RST}  ${DIM}${line}${RST}"
                ;;
            *model_management*|*Loading*|*Loaded*)
                echo -e "  ${BLUE}...${RST}  ${DIM}${line}${RST}"
                ;;
            *)
                echo -e "  ${DIM}${line}${RST}"
                ;;
        esac
    done || true

    if [ -f "${SESSION_FILE}.reboot" ]; then
        echo ""
        echo -e "  ${ARROW}  ${YELLOW}Manager requested restart...${RST}"
        echo ""
        sleep 1
        continue
    fi

    echo -e "  ${DIM}ComfyUI exited.${RST}"
    break
done
