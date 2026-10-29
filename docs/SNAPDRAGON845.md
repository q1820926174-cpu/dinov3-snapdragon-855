# OnePlus 6 / Snapdragon 845 qualification

[English](SNAPDRAGON845.md) | [简体中文](SNAPDRAGON845.zh-CN.md)

Measured on the connected OnePlus 6, confirmed `oneplus,enchilada` / `qcom,sdm845`, Adreno 630. Native Linux 7.3.0, Mesa 26.1.6, Vulkan loader 1.4.363, DRM render node `/dev/dri/renderD128`. Existing desktop, Waydroid, SSH and Docker services were preserved. No firmware, kernel, host driver, charging or clock policy was changed.

## Native model test

Exact same accepted model files, 224×224, strict Vulkan execution, 4 CPU threads, 48 same-input quality images. Speed uses the same 12-image subset repeated three times after three warmups, extracting both global and dense features.

| Backend | Mean ms | P95 ms | Inference FPS |
|---|---:|---:|---:|
| CPU FP32 | 206.07 | 210.21 | 4.85 |
| GPU packed FP16 / FP32 math | **147.27** | **150.68** | **6.79** |

GPU global mean cosine against official FP32 **0.999946**, P05 **0.999892**, minimum **0.999860**; dense mean **0.999883**. All six game groups passed the existing engineering feature gate. CPU was essentially identical at FP32 numerical precision. Feature preservation is not object detection accuracy.

The earlier 855 GPU measurement was 121.58 ms / 8.23 FPS with the same graph and input subset; this 845 measurement has about 21% higher latency. The phones have different kernels, Mesa versions, memory capacity, backgrounds and thermal states, so this is a deployment comparison, not an isolated chip-peak comparison. No paired same-graph 855 CPU RAW result is implied by the older CPU reference numbers.

## Actual Docker API

The host's Mesa 26.1 libraries were built against newer glibc than the released Bookworm image. Rather than copying incompatible host libraries or replacing host drivers, the optional `Dockerfile.mesa` installs Bookworm's open-source Mesa 22.3.6 **inside the container only**. Model/runner/API are the same as the released ARM64 base. Hardware was verified by the native `Turnip Adreno 630` log, not a software Vulkan driver.

| API mode | CPU | GPU |
|---|---:|---:|
| Sequential HTTP mean | **232.31 ms** | 246.01 ms |
| Sequential HTTP P95 | **242.64 ms** | 278.58 ms |
| Sequential throughput | **4.30 images/s** | 4.06 images/s |
| Batch4 throughput | 4.01 images/s | **4.29 images/s** |

Each single-image/concurrency benchmark used 16 requests cycling four real screenshots; batch4 used four requests. The test client was Python 3.11 + HTTPX in an isolated container using host networking to the server's actual published localhost port. Both backends loaded the same optimized model. The server remained UID 10001, read-only, nonprivileged, with only the DRM render device and model directory exposed.

Four API outputs against official FP32: GPU mean cosine **0.999952**, minimum **0.999943**; CPU approximately 1. Authentication, invalid-image handling, batch limits, similarity, single and batch embeddings, sequential and four-client operation passed.

**The compatibility Docker stack did not improve single-image latency with GPU:** CPU was slightly faster there, while GPU batch throughput was slightly higher. Native-host GPU was clearly faster than native CPU. Different driver versions and API execution paths prevent attributing all differences to hardware or conversion. Use the measured CPU container if single-request latency is the priority for this exact image; do not advertise the native 6.79 FPS as its HTTP throughput.

Temporary test containers and their test credentials were removed. The 855 production API was not modified. No permanent 845 service was left running by this test.

## Reproduce the DRM container layer

Build the repository API base or load the public v0.2.0 ARM64 image first, then:

```bash
docker tag dinov3-api:0.2.0-arm64 dinov3-api:local
export DINO_GPU_GID=$(stat -c '%g' /dev/dri/renderD128)
export DINO_MODEL_DIR="$PWD/gpu_conv_native"
docker compose -f compose.yaml -f compose.mesa.yaml up -d --build
```

This image uses a packaged driver and does not bind host Vulkan libraries. The default ICD path is ARM64 Freedreno; other architectures/GPUs need the correct `DINO_VK_ICD`. The host kernel still needs a supported DRM device and permissions. This does not support stock Android or the 855's KGSL path; use `compose.adreno.yaml` for the separately qualified 855.

## Power limitation and data

This test did **not** obtain the same ISYS×VSYS system-rail channel used on the 855. The OnePlus exposes battery and USB-charger telemetry, which must not be substituted for 855 system power without validating charging flow and losses. No 845 watt/joule or battery-life result is claimed.

`results/snapdragon845/native.json`, native timing CSVs and `docker-mesa22.json` retain the actual measurements, software versions and model/input hashes. Only short dynamic-clock tests were performed. No training, live game automation or new localization-quality claim is made.
