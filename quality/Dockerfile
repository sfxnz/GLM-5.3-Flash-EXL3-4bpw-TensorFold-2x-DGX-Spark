# Fidelity image: malaiwah/quant-fidelity-suite@cf36287c (MIT) on the NGC PyTorch base, for the hf-transformers
# hidden-state capture on one GB10 (sm_121) and the CPU compare. Upstream's own container/Dockerfile is amd64-only.
#   git init S && git -C S fetch --depth 1 --filter=blob:none https://github.com/malaiwah/quant-fidelity-suite <SUITE_REF>
#   (then sparse-checkout the paths below; quality/README.md has the exact commands)
#   docker build -f quality/Dockerfile --build-context suite=S -t glm53-fidelity:cf36287c-mcg1-stream quality/
# The suite is exported from the clone's git objects at SUITE_REF, so a dirty checkout cannot leak in.
# The exported set is upstream's bin/BUNDLE.txt (what its own image ships) plus the two CLI wrappers, LICENSE,
# fidelity-stats with its jointstd package, and exl3_scope.py (authors the --scope-file a quant capture needs).
# patches/0001-mcg-marker-rank1.patch and 0002-drop-private-shard-pages.patch are applied on top, in that order
# (quality/README.md, "Suite patches"): both captures run them.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

ARG SUITE_REF=cf36287c51a18e9361e009ca00a21eb8c0021ab1
# The base sets this to 1, and torch 2.13 then starts float32_matmul_precision at "high" (TF32 cuBLAS);
# ENV cannot unset a variable, and 0 restores torch's "highest" default (Context.h check_env).
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/fidelity/suite/bin:${PATH}

COPY requirements-capture.txt /tmp/requirements-capture.txt
COPY patches/0001-mcg-marker-rank1.patch patches/0002-drop-private-shard-pages.patch /tmp/
RUN python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements-capture.txt \
 && rm /tmp/requirements-capture.txt \
 && python3 -c "import torch, transformers, accelerate; assert torch.__version__.startswith('2.13.0a0+9186a08b2c'), torch.__version__; assert transformers.__version__ == '5.16.1'"

RUN --mount=type=bind,from=suite,target=/src \
    test "$(git -c safe.directory='*' -C /src rev-parse "$SUITE_REF^{commit}")" = "$SUITE_REF" \
 && mkdir -p /opt/fidelity/suite \
 && { git -c safe.directory='*' -C /src show "$SUITE_REF:bin/BUNDLE.txt" | grep -v -e '^#' -e '^$'; \
      printf '%s\n' LICENSE bin/fidelity-dataset bin/fidelity-stats bin/fidelity_stats.py \
        bin/jointstd/__init__.py bin/jointstd/canary.py bin/jointstd/chi2.py bin/jointstd/ngram.py \
        bin/jointstd/oracle.py bin/jointstd/protocol.py bin/jointstd/stats.py engines/tools/exl3_scope.py; } > /tmp/paths \
 && git -c safe.directory='*' -C /src archive "$SUITE_REF" -- $(cat /tmp/paths) | tar -x -C /opt/fidelity/suite \
 && test "$(find /opt/fidelity/suite -type f | wc -l)" = "$(sort -u /tmp/paths | wc -l)" \
 && git -C /opt/fidelity/suite apply /tmp/0001-mcg-marker-rank1.patch /tmp/0002-drop-private-shard-pages.patch \
 && echo "$SUITE_REF" > /opt/fidelity/suite/COMMIT \
 && (cd /tmp && sha256sum 0001-mcg-marker-rank1.patch 0002-drop-private-shard-pages.patch) > /opt/fidelity/suite/PATCHES \
 && rm /tmp/paths /tmp/0001-mcg-marker-rank1.patch /tmp/0002-drop-private-shard-pages.patch

LABEL fidelity.suite=malaiwah/quant-fidelity-suite fidelity.suite.revision=${SUITE_REF} \
      fidelity.suite.patches=0001-mcg-marker-rank1,0002-drop-private-shard-pages fidelity.transformers=5.16.1
WORKDIR /opt/fidelity/suite
