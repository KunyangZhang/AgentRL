#!/usr/bin/env bash
# Uses an existing CUDA-compatible PyTorch environment; never replaces CUDA wheels.
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON:-python}" - <<'PY'
import json, torch
assert torch.cuda.is_available(), 'CUDA PyTorch and an NVIDIA GPU are required'
p = torch.cuda.get_device_properties(0)
print(json.dumps({'gpu': p.name, 'vram_gib': p.total_memory/2**30, 'torch': torch.__version__, 'cuda': torch.version.cuda}))
assert p.total_memory >= 12*2**30, 'Initial profile requires >=12 GiB; 24 GiB recommended, not yet benchmarked'
PY
exec bash "$LAB_DIR/run.sh" train --config "$LAB_DIR/configs/lora.json" "$@"
