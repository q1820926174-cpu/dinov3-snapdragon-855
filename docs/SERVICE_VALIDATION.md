# Docker API qualification — 0.2.0

[English](SERVICE_VALIDATION.md) | [简体中文](SERVICE_VALIDATION.zh-CN.md)

## What ran

Real ARM64 Docker containers on the SCV41 / Snapdragon 855, with the repository's compiled native runner and the accepted optimized model. Both CPU and GPU used the **same optimized graph**, unlike some historical CPU/GPU comparisons in `MEASUREMENTS.md`.

The image uses Debian Bookworm, Python 3.11, FastAPI, uvloop/httptools and orjson. One model process is resident and warmed three times. CPU uses four native threads. GPU passes through `/dev/kgsl-3d0`, a read-only Turnip driver tree and the device group. The actual native log identified `Turnip Adreno (TM) 640`.

Actual HTTP requests crossed the Docker-published port using the phone's interface address; the client ran on the phone. This includes HTTP upload, JPEG decode, bicubic letterbox, model normalization/inference, queueing and JSON response. It does not measure a remote WAN, live capture or object detection.

## Throughput and latency

Each single-image benchmark uses 16 requests cycling through four real screenshots; each batch4 benchmark uses four requests, 16 images total. Dynamic clocks, short run, no new power measurement.

| Backend / mode | Mean HTTP latency | P95 HTTP latency | Throughput |
|---|---:|---:|---:|
| CPU, concurrency 1 | 183.44 ms | 194.43 ms | 5.41 images/s |
| GPU, concurrency 1 | **159.49 ms** | **170.20 ms** | **6.21 images/s** |
| CPU, concurrency 4 | 730.78 ms | 826.62 ms | 4.95 images/s |
| GPU, concurrency 4 | 584.71 ms | 650.90 ms | 6.17 images/s |
| CPU, batch of four | — | — | 5.04 images/s |
| GPU, batch of four | — | — | **6.42 images/s** |

GPU native global-feature extraction averaged 123.65 ms in the sequential run. HTTP latency is higher; the older 8.23-FPS RAW benchmark must not be advertised as API throughput. Concurrency four adds queueing and does not make a single GPU process execute four model instances in parallel.

## Accuracy and acceptance

Four service outputs were compared to the official FP32 ONNX model using exactly the API preprocessor. GPU normalized embedding mean cosine **0.999951677**, minimum **0.999937975**. CPU cosine was effectively 1 at numerical precision. This is a small feature-preservation check, not a labeled detection-quality evaluation.

Real-container checks passed: ready/live, required Bearer key, missing/malformed-key 401, four 768-dimensional normalized embeddings, batch4 base64 response, identical-image similarity near 1, invalid-image 422, nine-image batch 422, OpenAPI, sequential and four-client requests. GPU requested with no device/ICD failed startup rather than running CPU. Software-Vulkan rejection is also enforced in the native code; it was not a separate physical-driver test.

Docker inspect confirmed **UID/GID 10001:10001**, nonprivileged and read-only root filesystem. The existing phone applications were not stopped. Unit tests additionally cover streamed/declared body limits, bounded admission, child failure and cleanup, and response framing after a timed-out batch. Unit tests use a protocol double and are not the hardware evidence.

## Distribution and provenance

`results/service/acceptance.json` contains raw benchmark aggregates and quality checks. `results/service/image.json` records the exact accepted ARM64 image ID, archive hash and size. The downloadable image was built with `--target service-prebuilt` from the locally cross-compiled open-source runner; this is distinct from the full source-build target checked by CI.

The image archive was inspected: ARM64, nonroot default, no model files and no baked API key. Model and driver stay on read-only runtime mounts. Keys and private deployment paths are not published. The release contains no game media or Qualcomm SDK. See the workflow result for independent AMD64 source-image build status; ARM64 hardware acceptance does not by itself prove every other platform.

The independent [AMD64 full source-image build](https://github.com/q1820926174-cpu/dinov3-snapdragon-855/actions/runs/40850232246) passed, including native executable loading and nonroot image checks. [Linux API/protocol tests](https://github.com/q1820926174-cpu/dinov3-snapdragon-855/actions/runs/40850232371) also passed. AMD64 CI did not run a gated model or a physical GPU; those remain ARM64 device acceptance claims.
