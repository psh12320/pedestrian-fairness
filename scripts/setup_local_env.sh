#!/usr/bin/env bash
# Reproducible local setup for the pedestrian-fairness pipeline on macOS.
#
# Produces a Python 3.12 virtualenv at .venv-setup with traci, torch, numpy, and a
# prebuilt SUMO 1.28.0 (via the eclipse-sumo wheel -- NO compilation, NO admin pkg).
# Verified working: full SUMO+TraCI 3-junction corridor simulation and all unit tests.
#
# Usage:
#   bash scripts/setup_local_env.sh        # one-time install
#   source scripts/activate_local_env.sh   # per-shell: sets PATH/SUMO_HOME/PYTHONPATH
#
# Why eclipse-sumo and not Homebrew: the DLR Homebrew cask 404s (dead daily-build
# URL) and the DLR tap formula fails on modern Homebrew (removed cxxstdlib_check).
# The eclipse-sumo pip wheel ships the same prebuilt binaries and drops into the venv.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY=python3.12
command -v "$PY" >/dev/null || { echo "Need $PY on PATH (brew install python@3.12)"; exit 1; }

echo "== creating venv (.venv-setup) with $($PY --version) =="
"$PY" -m venv .venv-setup
.venv-setup/bin/pip install --quiet --upgrade pip

echo "== installing python deps (traci, numpy, torch) =="
.venv-setup/bin/pip install --quiet "traci>=1.20,<2" numpy torch

echo "== installing prebuilt SUMO (eclipse-sumo wheel) =="
.venv-setup/bin/pip install --quiet eclipse-sumo

SUMO_PKG="$ROOT/.venv-setup/lib/python3.12/site-packages/sumo"
echo "== verifying =="
PATH="$ROOT/.venv-setup/bin:$PATH" sumo --version | head -1
echo "SUMO_HOME would be: $SUMO_PKG"
echo
echo "Done. Activate a shell with:  source scripts/activate_local_env.sh"
