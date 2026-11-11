#!/usr/bin/env bash
# ==============================================================================
# Script de ejecución de pipeline para Linux / macOS (Bash)
# Tarea: Clasificación Supervisada de Texto (Corpus Reddit C-SSRS, N=500)
# ==============================================================================
set -e

echo "[INFO] Iniciando pipeline de Clasificación Supervisada..."

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] No se encontró Python en el PATH." >&2
    exit 1
fi

$PYTHON_CMD main.py "$@"
echo "[OK] Pipeline completado con éxito."
