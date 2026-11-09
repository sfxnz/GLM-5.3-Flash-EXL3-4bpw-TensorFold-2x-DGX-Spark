ARG TF_VERSION=1.2.5
ARG TF_SHA256=1a0192b28d6ab6ec2749f3ff25515379dba4895171e781b30f810afc971ae4f8

# nvidia/cuda:13.0.0-base-ubuntu24.04 (arm64). CUDA 13.0 matches the R580 driver, and its apt source is
# NVIDIA's sbsa repo, which carries cuBLAS 13 and NCCL 2 without pulling a 24 GB framework image.
FROM nvidia/cuda:13.0.0-base-ubuntu24.04@sha256:6e43a6b02e5f16e4a715953be8b29c40acbe84aed11a61357ef0fc899967fbd9 AS base

# Replace with --build-context downloads=DIR when DIR already holds these files; every consumer checks its sha256.
FROM scratch AS downloads
ARG TF_VERSION
ADD https://github.com/ashhart/TensorFold/releases/download/v${TF_VERSION}/tensorfold-${TF_VERSION}-linux-aarch64.tar.gz /
ADD https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-official_3.1.4.1_arm64.deb /
ADD https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz /

# TensorFold refuses any JPEG/WebP decoder whose embedded version is not exactly libjpeg-turbo 3.1.4.1 / libwebp 1.6.0,
# and Ubuntu 24.04 ships older ones.
FROM base AS codecs
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev make && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=downloads,target=/dl \
    printf '%s  %s\n' \
      d3f9d7ddd998758e1850b37153a7904bdcdeb8cf54a782c772780b0bc50dd553 /dl/libjpeg-turbo-official_3.1.4.1_arm64.deb \
      e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564 /dl/libwebp-1.6.0.tar.gz | sha256sum -c - \
 && mkdir -p /out /src/jpeg \
 && dpkg-deb -x /dl/libjpeg-turbo-official_3.1.4.1_arm64.deb /src/jpeg \
 && cp -L /src/jpeg/opt/libjpeg-turbo/lib64/libturbojpeg.so.0 /out/ \
 && tar -xzf /dl/libwebp-1.6.0.tar.gz -C /src && cd /src/libwebp-1.6.0 \
 && ./configure --enable-libwebpdecoder --disable-static \
 && make -C sharpyuv -j"$(nproc)" && make -C src -j"$(nproc)" \
 && cp -L src/.libs/libwebpdecoder.so.3 /out/

FROM base
ARG TF_VERSION
ARG TF_SHA256
# NCCL 2.30.7 is the version upstream's GLM TP receipts ran; ibverbs-providers carries the mlx5 provider for RoCE gathers.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcublas-13-0=13.1.1.3-1 libnccl2=2.30.7-1+cuda13.3 libibverbs1 ibverbs-providers \
 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=bind,from=downloads,target=/dl \
    echo "${TF_SHA256}  /dl/tensorfold-${TF_VERSION}-linux-aarch64.tar.gz" | sha256sum -c - \
 && mkdir /opt/tensorfold \
 && tar -xzf /dl/tensorfold-${TF_VERSION}-linux-aarch64.tar.gz -C /opt/tensorfold --strip-components=1 \
 && cd /opt/tensorfold && sha256sum -c --quiet SHA256SUMS
# ../lib from the binary is the first place the server looks for libtfgrammar and both codecs.
COPY --from=codecs /out/ /opt/tensorfold/lib/
ENV PATH=/opt/tensorfold/bin:${PATH}
LABEL tensorfold.version=${TF_VERSION} \
      tensorfold.archive=tensorfold-${TF_VERSION}-linux-aarch64.tar.gz \
      tensorfold.archive.sha256=${TF_SHA256}
