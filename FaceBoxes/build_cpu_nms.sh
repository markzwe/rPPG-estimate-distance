SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$SCRIPT_DIR/utils"
"$PYTHON_BIN" build.py build_ext --inplace
