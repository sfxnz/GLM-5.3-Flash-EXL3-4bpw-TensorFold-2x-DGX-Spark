#!/usr/bin/env bash
# Usage: quality/verify-image.sh <dir holding a GLM-5.3-Flash config.json>
#   CPU-only checks of glm53-fidelity (no GPU, no network): pins, TF32 default, the suite CLIs, and a 2-layer
#   random-weight forward of the real architecture through the suite's import path, which must not load cutlass.
set -euo pipefail
cfg=$(realpath "${1:?usage: verify-image.sh <dir with config.json>}")
image=${IMAGE:-glm53-fidelity:cf36287c-mcg1-stream}
run=(docker run --rm --network none --user 1000:1000 -e HOME=/tmp -v "$cfg/config.json:/cfg/config.json:ro")
"${run[@]}" --entrypoint bash "$image" -c '
set -e
echo "suite $(cat /opt/fidelity/suite/COMMIT), $(find /opt/fidelity/suite -type f | wc -l) files"
python3 -c "import torch, transformers, accelerate, numpy, os
print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"arch\", torch.cuda.get_arch_list())
print(\"transformers\", transformers.__version__, \"accelerate\", accelerate.__version__, \"numpy\", numpy.__version__)
print(\"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=%s allow_tf32=%s float32_matmul_precision=%s\" % (os.environ.get(\"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE\"), torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()))"
pip check || true
fidelity-dataset --help | head -2
fidelity-stats --help | head -1
python3 engines/tools/hf_capture.py --help | head -1
'
"${run[@]}" -e PYTHONPATH=/opt/fidelity/suite/bin:/opt/fidelity/suite/engines/tools --entrypoint python3 "$image" -c '
import sys, torch, transformers
import hf_capture, layer_outer, exl3hf_surface, exl3_scope
from fidelity import dscompare, dsvalidate
cfg = transformers.AutoConfig.from_pretrained("/cfg")
t = cfg.text_config
t.num_hidden_layers, t.first_k_dense_replace, t.n_routed_experts, t.n_group, t.topk_group = 2, 1, 8, 1, 1
t.num_experts_per_tok = min(t.num_experts_per_tok, 8)
torch.manual_seed(0)
model = getattr(transformers, cfg.architectures[0])(cfg).to(torch.bfloat16).eval()
with torch.no_grad():
    logits = model(input_ids=torch.randint(0, 1000, (1, 64))).logits
print("forward", type(model).__name__, tuple(logits.shape), "finite", bool(torch.isfinite(logits.float()).all()))
loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("cutlass", "nvidia_cutlass_dsl", "kernels", "flash_attn"))
print("cutlass/kernels modules loaded:", loaded or "none")
sys.exit(1 if loaded else 0)
'
