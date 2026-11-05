# Quantizer image: ExLlamaV3 v1.6.6 built for GB10 (sm_121a). Encodes only; never serve from it.
#   git clone https://github.com/turboderp-org/exllamav3 /path/to/exllamav3
#   docker build -f quant/Dockerfile --build-context exllamav3=/path/to/exllamav3 -t glm53-quant:exl3-1.6.6 quant/
# The build exports the pinned commit from that clone's git objects (git archive), so a dirty or
# moved checkout cannot leak in, and it fails unless the commit is present.
# v1.6.6 builds natively on aarch64 (upstream 181eea1e guards the x86 AVX paths), so no source patch.
# llguidance is left out: only the generator's grammar filter imports it, never convert.py.
# The base sets TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 (forced TF32 matmuls). ENV cannot unset it, so run
# convert.py under `env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` (quant/convert.sh does).
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

ARG EXLLAMAV3_REF=bd1f519172406a9560bd5c90e61000691bb69a07
ARG MAX_JOBS=8
RUN --mount=type=bind,from=exllamav3,target=/src \
    test "$(git -c safe.directory='*' -C /src rev-parse "$EXLLAMAV3_REF^{commit}")" = "$EXLLAMAV3_REF" \
 && mkdir /opt/exllamav3 \
 && git -c safe.directory='*' -C /src archive "$EXLLAMAV3_REF" | tar -x -C /opt/exllamav3 \
 && python3 -m pip install --no-cache-dir --no-deps --retries 10 marisa-trie==1.4.1 \
 && MAX_JOBS="$MAX_JOBS" TORCH_CUDA_ARCH_LIST=12.1a python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/exllamav3 \
 && rm -rf /opt/exllamav3/build \
 && echo "$EXLLAMAV3_REF" > /opt/exllamav3/COMMIT \
 && python3 -c "import importlib.metadata as md, exllamav3, exllamav3_ext; print('exllamav3', md.version('exllamav3'))"

COPY smoke_quantize.py /opt/glm53-quant/smoke_quantize.py
WORKDIR /opt/exllamav3
