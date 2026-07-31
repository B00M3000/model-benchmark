#!/usr/bin/env bash
# Start the benchmark server.
#
# On the Jetson, pin the clocks first or the latency numbers are noise:
#   sudo nvpmodel -m 0 && sudo jetson_clocks
set -euo pipefail

cd "$(dirname "$0")/.."

HOST="${BENCH_HOST:-0.0.0.0}"
PORT="${BENCH_PORT:-8000}"

PYTHON="${PYTHON:-python3}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi

echo "Starting on http://${HOST}:${PORT}"
"$PYTHON" -c "
from benchmark.models.registry import resolve_backend
from benchmark.config import load_config
backend = resolve_backend(load_config())
print(f'Backend: {backend}')
if backend == 'mock':
    print('  WARNING: Jetson libraries not importable -- latencies will be synthetic.')
"

exec "$PYTHON" -m uvicorn benchmark.server:app --host "$HOST" --port "$PORT"
