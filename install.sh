#!/usr/bin/env bash
set -euo pipefail

# Operational paths complying with XDG Base Directory Specification
GLIM_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/glim"
MODEL_DIR="$GLIM_DIR/models"
BIN_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
MODEL_NAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/$MODEL_NAME"

echo "[*] Initializing Glim deployment environment..."
mkdir -p "$MODEL_DIR" "$BIN_DIR"

# 1. Resolve host platform architecture for binary target selection
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ]; then ARCH="amd64"; fi

BINARY_URL="https://github.com/yourusername/glim/releases/latest/download/glim_${OS}_${ARCH}"

# 2. Fetch the minimal core compiled engine
echo "[*] Downloading Glim runtime architecture asset..."
curl -sSL -o "$BIN_DIR/glim" "$BINARY_URL"
chmod +x "$BIN_DIR/glim"

# 3. Check for or fetch the heavy model data matrix
if [ ! -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "[*] Downloading Qwen2.5-1.5B Engine Matrix (~1.1GB)..."
    # Utilizing curl's resume capability to tolerate dropped connections in production pipes
    curl -L -C - -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
else
    echo "[+] Validated existing model matrix cache. Skipping payload transfer."
fi

echo "[+] Deployment successful. Glim engine ready at $BIN_DIR/glim"