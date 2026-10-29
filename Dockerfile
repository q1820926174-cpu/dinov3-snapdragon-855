# syntax=docker/dockerfile:1
FROM debian:bookworm-slim AS runtime-base
LABEL org.opencontainers.image.source="https://github.com/q1820926174-cpu/dinov3-snapdragon-855" \
      org.opencontainers.image.title="DINOv3 ncnn API" \
      org.opencontainers.image.licenses="MIT AND BSD-3-Clause" \
      org.opencontainers.image.version="0.2.0"
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv ca-certificates libvulkan1 libstdc++6 libgomp1 libzstd1 \
    libxcb-dri3-0 libx11-xcb1 libxcb-present0 libxcb-xfixes0 libxcb-sync1 \
    libxcb-randr0 libxcb-shm0 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
COPY requirements-service.txt /tmp/requirements-service.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-service.txt \
    && rm /tmp/requirements-service.txt \
    && groupadd --gid 10001 dino \
    && useradd --uid 10001 --gid 10001 --home-dir /app --shell /usr/sbin/nologin dino
WORKDIR /app
COPY service /app/service
COPY LICENSE third_party/ncnn-LICENSE.txt /usr/share/licenses/dinov3-api/
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DINO_BACKEND=cpu DINO_MODEL_PATH=/models/model DINO_NUM_THREADS=4 \
    DINO_MAX_QUEUE=8 DINO_TIMEOUT_SECONDS=30 OPENBLAS_NUM_THREADS=1
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=3s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready',timeout=2)"
CMD ["python","-m","uvicorn","service.app:app","--host","0.0.0.0","--port","8000","--workers","1","--limit-concurrency","32","--timeout-keep-alive","5","--no-access-log"]

# Fast deployment when a trusted, architecture-matched open-source binary is
# already built. No weights or vendor driver are copied into either target.
FROM runtime-base AS service-prebuilt
USER root
COPY .docker-build/dinov3_embed_server /usr/local/bin/dinov3_embed_server
RUN chmod 755 /usr/local/bin/dinov3_embed_server
USER 10001:10001

FROM debian:bookworm-slim AS native-build
ARG NCNN_COMMIT=9fc8106e6634a60f39945d63e5ed2dd0704ac9fc
ARG BUILD_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake ninja-build g++ libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git init /src/ncnn && git -C /src/ncnn remote add origin https://github.com/Tencent/ncnn.git \
    && git -C /src/ncnn fetch --depth 1 origin ${NCNN_COMMIT} \
    && git -C /src/ncnn checkout --detach FETCH_HEAD \
    && git -C /src/ncnn submodule update --init --recursive --depth 1
COPY patches/ncnn-scv41.patch /tmp/ncnn.patch
RUN git -C /src/ncnn apply /tmp/ncnn.patch \
    && cmake -G Ninja -S /src/ncnn -B /build/ncnn -DCMAKE_BUILD_TYPE=Release \
       -DNCNN_VULKAN=ON -DNCNN_SHARED_LIB=OFF -DNCNN_BUILD_TOOLS=OFF \
       -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF \
       -DCMAKE_INSTALL_PREFIX=/opt/ncnn \
    && cmake --build /build/ncnn -j ${BUILD_JOBS} && cmake --install /build/ncnn
COPY CMakeLists.txt ncnn_embed_server.cpp ncnn_runner.cpp operator_bench.cpp /src/app/
RUN cmake -G Ninja -S /src/app -B /build/app -DCMAKE_BUILD_TYPE=Release \
        -Dncnn_DIR=/opt/ncnn/lib/cmake/ncnn \
    && cmake --build /build/app --target dinov3_embed_server -j ${BUILD_JOBS}

FROM runtime-base AS service
COPY --from=native-build /build/app/dinov3_embed_server /usr/local/bin/dinov3_embed_server
