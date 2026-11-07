set -e
cd "$(dirname "$0")/.."
PYTHONPATH=src .venv/bin/python -m molecula.collect --out data "$@"
