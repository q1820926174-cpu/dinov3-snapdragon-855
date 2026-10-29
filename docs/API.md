# Docker & API guide

[English](API.md) | [简体中文](API.zh-CN.md)

## Features and scope

One warm ncnn process per container, asynchronous HTTP handling, bounded FIFO admission, three startup warmups, cancellation-safe native IPC, upload/pixel limits, optional Bearer authentication, readiness/liveness, and OpenAPI at `/docs`.

The API produces **768-dimensional L2-normalized global image embeddings**, or cosine similarity between two images. It does not provide a trained detector, dense feature maps or the historical OmniMatch sliding-window pipeline. One HTTP batch groups images but executes them sequentially on one native worker; it is not tensor batching or concurrent GPU model instances. A small concurrent queue can improve utilization while maintaining bounded memory.

Backends: `cpu` (FP32, configurable OpenMP threads) or `gpu16s` (ncnn Vulkan, FP16 packed storage + FP32 math). Unsupported Vulkan layers, missing GPU devices and CPU software Vulkan devices fail explicitly. There is no automatic CPU fallback, CUDA backend, NPU backend or full-FP16 arithmetic mode.

## Build from source

Obtain and convert the authorized weights as in README. The directory mounted to `/models` must contain `model.param` and `model.bin`. Weights and vendor drivers are never embedded in the container image.

```bash
docker build --target service -t dinov3-api:local .
export DINO_MODEL_DIR="$PWD/gpu_conv_native"
docker compose up -d
curl http://127.0.0.1:8000/health/ready
```

The Dockerfile builds the pinned, patched ncnn and native runner in a build stage, then copies only the executable into a Python runtime. It builds for the Docker builder's architecture. Native ARM64 builds are preferred; `buildx --platform linux/arm64` on an x86 machine additionally requires a configured cross-platform/QEMU builder. Hardware acceleration requires an actual host GPU at runtime; an emulated build does not emulate Adreno acceleration.

For an already-built trusted architecture-matched executable:

```bash
mkdir -p .docker-build
cp build/dinov3_embed_server .docker-build/dinov3_embed_server
docker build --target service-prebuilt -t dinov3-api:local .
```

The prebuilt path uses the same API/runtime and is the ARM64 image path qualified on the phone. Build your executable with the repository's pinned patched ncnn; arbitrary binaries do not establish the no-fallback guarantee.

## Snapdragon 855 Linux / Turnip

```bash
export DINO_GPU_GID=$(stat -c '%g' /dev/kgsl-3d0)
export DINO_MODEL_DIR="$PWD/gpu_conv_native"
docker compose -f compose.yaml -f compose.adreno.yaml up -d --build
```

This maps `/dev/kgsl-3d0`, adds its numeric group, and mounts the existing compatible Turnip installation read-only. The example expects `/opt/scv41-mesa-kgsl` with its ARM64 ICD JSON. Configure paths for your phone. The container remains nonroot, read-only and nonprivileged; do not use `--gpus all` as a substitute for KGSL passthrough.

`compose.vulkan.yaml` is an **unqualified generic DRM example** for `/dev/dri`. Set `DINO_GPU_GID`, `DINO_VK_DRIVER_DIR` and `DINO_VK_ICD` (a path inside the mounted container tree); the ICD's library paths and dependencies must be valid in the image. It does not install host drivers or guarantee compatibility with every GPU. Native Linux/Turnip qualification is not an Android APK claim.

For the separately tested **OnePlus 6 / Snapdragon 845 DRM path**, use `Dockerfile.mesa` and `compose.mesa.yaml`. This optional layer installs open-source Mesa packages inside the container, avoiding a host-driver/glibc mismatch. Set `DINO_GPU_GID` to the numeric group of `/dev/dri/renderD128`; see [845 results and limitations](SNAPDRAGON845.md). The tested Mesa 22.3 compatibility image has slightly worse single-image GPU HTTP latency than CPU; it must not be advertised using native-host timing.

## Request examples

```bash
curl -F file=@image.jpg http://127.0.0.1:8000/v1/embeddings
curl -F file=@image.jpg 'http://127.0.0.1:8000/v1/embeddings?encoding_format=base64'
curl -F files=@a.jpg -F files=@b.jpg http://127.0.0.1:8000/v1/embeddings/batch
curl -F image_a=@a.jpg -F image_b=@b.jpg http://127.0.0.1:8000/v1/similarity
```

```python
import os, requests
headers = {"Authorization": "Bearer " + os.environ["DINO_API_KEY"]}
with open("image.jpg", "rb") as image:
    response = requests.post("http://127.0.0.1:8000/v1/embeddings",
                             files={"file": image}, headers=headers, timeout=30)
response.raise_for_status()
vector = response.json()["data"][0]["embedding"]  # 768 normalized floats
```

`encoding_format=base64` returns base64-encoded **little-endian float32**. Default `float` returns a JSON array. The response includes `backend`, `dimensions`, `normalized`, per-image `native_ms`, and queue/processing timings. Processing includes image decode/resize and native calls; client wall time additionally includes HTTP upload, response encoding and transport. `native_ms` includes RGB normalization and global feature extraction, not the earlier two-output RAW runner timing.

## Configuration and operation

| Environment | Default | Meaning |
|---|---|---|
| `DINO_BACKEND` | `cpu` | `cpu` or `gpu16s`, no automatic fallback |
| `DINO_MODEL_PATH` | `/models/model` | Prefix of `.param` / `.bin` |
| `DINO_EXECUTABLE` | `/usr/local/bin/dinov3_embed_server` | Native executable |
| `DINO_NUM_THREADS` | `4` | Native CPU thread count, 1–64 |
| `DINO_MAX_QUEUE` | `8` | Pending requests; one active request is additional |
| `DINO_TIMEOUT_SECONDS` | `30` | Request deadline / individual native IO deadline |
| `DINO_STARTUP_TIMEOUT_SECONDS` | `60` | Native initial readiness header deadline |
| `DINO_MAX_BODY_BYTES` | `8388608` | Entire multipart/body size, including batch overhead |
| `DINO_MAX_BATCH` | `8` | Maximum images in a logical batch |
| `DINO_MAX_IMAGE_PIXELS` | `20000000` | Decoded source image pixel limit |
| `DINO_API_KEY` | empty | Optional Bearer key; set it for remote clients |

Compose binds `127.0.0.1:8000` by default. Set `DINO_BIND_ADDRESS` / `DINO_PORT` explicitly to expose another interface. For remote deployment set a key and use an authenticated TLS reverse proxy. Keys belong in a local environment/secret store, never Git or an image. Health endpoints remain unauthenticated and contain no key or model file path.

HTTP errors: `401` authentication; `413` request body limit; `422` invalid/unsupported image or batch; `503` queue full/worker unavailable (`Retry-After: 1` for a full queue); `504` request deadline. A timed-out request does not cancel native pipe reads, preserving subsequent response framing. A native crash marks health unhealthy and returns 503; restart the container after resolving the cause. Docker's health flag alone does not restart an otherwise running unhealthy process.

Use **one Uvicorn worker per container/GPU** to avoid duplicating weights and queues. The image uses a 32-connection concurrency limit; excess connections can also receive Uvicorn's 503. Scale with separate devices/containers rather than setting many workers against the same small GPU. Models stay resident; replacing mounted files requires restart. Do not mutate model files in place while serving.

## Tests and benchmarking

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/benchmark_api.py --url http://127.0.0.1:8000 --images a.jpg b.jpg --requests 16 --concurrency 4
```

The benchmark reads `DINO_API_KEY` from its environment and reports actual HTTP latency, throughput and native execution time without printing the key. Unit tests use a protocol test double and make no hardware/performance claim. Real ARM64 Docker/Turnip acceptance is recorded separately in `results/service/` when available. Historical model/application/power results remain in `docs/MEASUREMENTS.md`; do not relabel them as new API throughput.

References: [Docker multi-stage builds](https://docs.docker.com/get-started/docker-concepts/building-images/multi-stage-builds/), [FastAPI containers and worker memory](https://fastapi.tiangolo.com/deployment/docker/).
