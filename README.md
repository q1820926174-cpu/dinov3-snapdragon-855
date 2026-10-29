# DINOv3 on Snapdragon 855 / Adreno 640

[English](README.md) | [简体中文](README.zh-CN.md)

Optimized **DINOv3 ConvNeXt-Tiny** conversion, persistent Docker/HTTP inference, and measured accuracy, application latency and system energy on a Samsung S10 SCV41 running native Linux + Mesa Turnip.

## Docker & hardware-accelerated API

A warm native ncnn worker serves asynchronous HTTP requests with bounded admission, deadlines, upload limits, optional Bearer authentication, readiness/liveness and bilingual OpenAPI at `/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/embeddings` | One image → 768 normalized global features |
| `POST /v1/embeddings/batch` | Multiple images, JSON floats or base64 float32 |
| `POST /v1/similarity` | Two images → cosine similarity |
| `GET /v1/models` | Model and runtime information |
| `GET /health/ready` | Warm native worker readiness |

Backends: CPU FP32 or **ncnn Vulkan GPU** with FP16 packed storage / FP32 math. GPU requests fail explicitly when unavailable; software CPU Vulkan devices are rejected. No automatic CPU fallback, CUDA, NPU or full-FP16 arithmetic backend is claimed.

The API extracts global features, not object detections. A logical HTTP batch executes sequentially on one native worker, not tensor batching. Model weights and vendor drivers are mounted read-only and are not included in images.

```bash
# First obtain authorized weights and convert them as below.
docker compose up -d --build                         # CPU
# Snapdragon 855 native Linux + an existing compatible Turnip installation:
export DINO_GPU_GID=$(stat -c '%g' /dev/kgsl-3d0)
docker compose -f compose.yaml -f compose.adreno.yaml up -d --build
curl -F file=@image.jpg http://127.0.0.1:8000/v1/embeddings
```

See **[English API/Docker guide](docs/API.md)** / **[中文 API/Docker 文档](docs/API.zh-CN.md)** for authentication, limits, driver passthrough, image architecture, prebuilt-binary packaging, error handling and benchmarking. The default listener is localhost. Generic `/dev/dri` Vulkan Compose is an unqualified example; native Linux/Turnip results do not establish Android compatibility.

**[0.2.0 ARM64 image download](https://github.com/q1820926174-cpu/dinov3-snapdragon-855/releases/tag/v0.2.0)** · [Real-container qualification](docs/SERVICE_VALIDATION.md). On the 855: GPU mean single-image HTTP latency **159.49 ms**, batch4 **6.42 images/s**, four-image mean FP32-reference cosine **0.999951677**. These are actual API measurements, distinct from the historical RAW results below.

## Obtain and convert the model

Accept the applicable conditions and obtain access on the [official model page](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m). This repository does not distribute weights or grant rights to them.

```bash
python -m venv .venv
# Activate your environment, then:
python -m pip install -r requirements.txt
hf auth login
hf download facebook/dinov3-convnext-tiny-pretrain-lvd1689m --local-dir models/dinov3-convnext-tiny
python export_gpu_conv_native.py --model models/dinov3-convnext-tiny --out gpu_conv_native
```

Outputs: `gpu_conv_native/model.param` and `model.bin`. Weights remain FP32 on disk; packed half storage is a runtime option. The converter checks rewritten PyTorch / official FP32 / TorchScript parity; physical-device validation is still required.

## Measured conversion result

Same official weights, no retraining or smaller replacement model. Native LayerNorm/GELU plus 1×1 Conv MLP reduces the graph to **188 nodes, zero Gemm**. The Vulkan GELU implementation uses a tanh approximation.

| Deployment path, 224×224 | Mean latency | Inference FPS |
|---|---:|---:|
| Optimized GPU | **121.58 ms** | **8.23** |
| Previous decomposed GPU graph | 226.40 ms | 4.42 |
| Previous graph CPU FP32 reference | 185.05 ms | 5.40 |

The optimized GPU has 1.86× throughput / 46.3% lower latency than the old GPU graph. Across 48 engineering images from six games: global mean cosine **0.999946**, P05 **0.999892**, dense mean **0.999899**. These are feature-preservation metrics, not detection mAP. The images were used in engineering tuning, not a new blind test. Full-FP16 arithmetic failed quality; the all-native Gemm control was slower and is not recommended.

The native RAW runner excludes loading, warmup, file IO and input layout conversion. These values are **not HTTP API throughput**. Updated service acceptance is recorded separately in `results/service/` and the API guide.

## Real application and energy

The original [OmniMatch CV](https://github.com/drtiwari/OmniMatch_CV) API, sliding windows, FAISS and NMS were run on the phone with this feature provider. Each 806×640 screenshot required 120 windows.

| Synchronized power test, two screenshots/backend | Optimized GPU | CPU FP32 reference |
|---|---:|---:|
| Mean system DC-rail power | **3.62 W** | 4.64 W |
| Complete screenshot request | **14.81 s** | 25.40 s |
| System energy / screenshot | **53.65 J** | 117.91 J |

GPU system energy was 54.5% lower in this short workload. These are on-board system-rail estimates with the screen off and background services running, not chip-only or calibrated wall-meter power. They are not long-run battery-life claims.

**Application localization failed.** Four earlier real screenshots produced identical CPU/GPU boxes, but fixed 64×64 output windows mismatched ~40×40 annotations and the negative case had false detections. Feature similarity does not establish usable detection quality. API acceptance used an actual in-process ASGI route; no Streamlit UI, TCP network or live-game capture was measured in that historical experiment.

## Reproduce and inspect

- [Build / device reproduction](docs/REPRODUCE.md): pinned patched ncnn, ARM64 cross builds, preprocessing and the original application integration.
- [Measurement scope and failures](docs/MEASUREMENTS.md): sample counts, baseline differences, power units and unsuccessful controls.
- [Publication validation](docs/PUBLICATION_VALIDATION.md): the portable converter reproduced the exact phone-tested model hashes.
- `results/`: sanitized metrics, real application responses, power telemetry and timing markers; no images or feature binaries.
- `scripts/prepare_app_cases.py`: reconstruct the fixed public test cases, checking source hashes; images remain ignored by Git.
- `scripts/summarize_power.py`: independently recompute request-integrated power/energy.
- `scripts/benchmark_api.py`: actual HTTP benchmark; credentials are read from environment and never printed.

The tested ncnn commit is `9fc8106e6634a60f39945d63e5ed2dd0704ac9fc`; apply `patches/ncnn-scv41.patch`. Native runners can also be built without Docker. Neither the Qualcomm SDK nor a ready-to-flash phone system is provided.

## NPU status

This is a **GPU deployment**. Whole-model HTA compilation and whole-model W8A8 quality did not pass. A stem-only NPU/GPU prototype covered about **0.318% of original MACs** and missed overall accuracy/speed targets. No high-NPU-occupancy backbone or successful NPU/GPU collaboration speedup is claimed.

## License

Original code: [MIT](LICENSE). Upstream ncnn patch context retains BSD-3-Clause. Models, third-party applications, datasets and SDKs retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). No credentials, weights, game media, proprietary SDK or driver binaries are published.
