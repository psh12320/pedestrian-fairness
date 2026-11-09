# Source this file to configure the current shell for the local pipeline.
#   source scripts/activate_local_env.sh
# Assumes scripts/setup_local_env.sh has already created .venv-setup.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export SUMO_HOME="$_ROOT/.venv-setup/lib/python3.12/site-packages/sumo"
export PATH="$_ROOT/.venv-setup/bin:$PATH"
export PYTHONPATH="$_ROOT/src:$SUMO_HOME/tools"
# Single-threaded math matches the cluster protocol and keeps runs deterministic.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Activated: $(sumo --version 2>/dev/null | head -1), $(python --version 2>&1)"
echo "SUMO_HOME=$SUMO_HOME"
