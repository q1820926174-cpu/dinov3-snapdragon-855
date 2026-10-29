# Reproduction notes

## Host export

Python 3.11, PyTorch CPU, Transformers 4.57.6 and PNNX 20261020 were used for conversion. The public converter accepts `--model`, `--out`, and optional `--pnnx`; it supports Windows and POSIX executable names. Its output directory is created recursively.

Download the gated official model yourself after obtaining authorization. Do not put a Hugging Face token in source files or command examples. The converter's PyTorch parity check is necessary but does not replace device verification.

## ncnn and ARM64

The tested dependency is pinned in README. Apply `patches/ncnn-scv41.patch` only to that clean revision; do not overwrite an unrelated installed ncnn checkout. The patch restores the actual reported Turnip shaderFloat16 feature, provides an opt-in historical Gemm compatibility experiment, and rejects attempted CPU fallback under `SCV41_NCNN_STRICT_GPU`. The recommended Conv-MLP graph has no Gemm and does not require `SCV41_NCNN_SAFE_GEMM`.

For a cross build, configure both ncnn and this project with your ARM64 toolchain. For example, a toolchain can set:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
# Set CMAKE_SYSROOT and dependency search roots for your target as needed.
```

The target needs a compatible C++ runtime, Vulkan loader and Turnip/KGSL driver. The measurement phone used Linux 4.14.190 and Mesa Turnip 26.2.5. Driver installation, kernel bring-up, flashing and USB configuration are outside this repository. Do not infer Android compatibility from the native Linux result.

The historical driver location was `/opt/scv41-mesa-kgsl`; change environment variables to your actual installation. Do not copy another machine's SDK or proprietary driver files into this repository.

## Input and feature validation

`prepare_multigame_calibration.py` uses square letterbox padding RGB 114, float32 scaling by 255, mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)`. Use the same preprocessor for reference and device. `dinov3_ncnn` reads normalized NHWC RAW; the persistent `dinov3_embed_server` instead reads 224 RGB uint8 and performs normalization itself. Do not normalize its input twice.

Global output is 768 floats. Dense output at 224 is 768×7×7, the native final-stage stride is 32. `evaluate_features.py` compares global/dense feature preservation; it does not compute object-detection accuracy.

## Application sample

```bash
python scripts/prepare_app_cases.py
```

This fetches the pinned upstream OmniMatch source and seven public source JPEG files, verifies their hashes, and reconstructs the three gallery crops and four annotated cases. Images remain ignored by Git. It does not download model weights or install a phone runtime. Network and source permissions are required.

Install `requirements-app.txt` in an isolated **ARM64 phone Python environment**, not over unrelated global services. Copy this repository's `real_app` directory and required model/ARM64 executable artifacts to the phone. The measured layout used the repository parent of `real_app` for:

- `dinov3_embed_server`: ARM64 executable;
- `dinov3_conv_native.param/.bin`: accepted GPU graph;
- `dinov3_gpu_adaptive.param/.bin`: historical CPU FP32 reference graph.

The two graphs contain the same base model weights but differ in operator lowering. Historical CPU/GPU timing is a comparison of these actual deployment paths, not a best-possible CPU kernel claim. The private model files are intentionally absent; the legacy reference can be reconstructed with `export_snpe_compatible.py`, the same calibration and historical converter, but its exact original artifacts are not bundled. Record any different baseline you choose.

`real_app/run_phone.py` uses the original upstream API through an in-process ASGI TestClient, a fixed CPU-reference gallery, and both backends. It expects the prepared data and correct model files; it will not silently substitute a different model. It does not launch a public HTTP server or Streamlit UI.

## Power measurement

The supplied monitor is specific to this Samsung/MAX77705 driver ABI. In this ABI `battery/power_now` is **ISYS in mA**, not μW. The read-only 0xB1 register gives VSYS mV after `raw*15625//100000`. Measure `ISYS*VSYS/1e6` watts, not USB input or battery charge power. Revalidate this before using another device or kernel.

`real_app/power_app_adapter.py` copies the measured adapter definitions; `power_worker.py` runs GPU0→CPU0→CPU1→GPU1, 8-second idle intervals and 120 windows/request. `power_monitor.py` samples at approximately 2 Hz and needs permission to read fuel-gauge registers. It does not write charging, frequency, screen or thermal controls. A 45°C battery stop and child-process cleanup are included.

The device-specific collector expects `/run/scv41-npu-runtime/lock` to be held by the launcher. Root sampler / dev-user application separation and `/opt/scv41-mesa-kgsl` paths describe the measured phone, not portable defaults. The original launcher was intentionally not published because it depended on a private SSH helper; use your own SSH configuration and hold the compute lock so concurrent workloads do not contaminate power readings.

Public results contain sanitized historical records. Hashes inside records refer to the originally measured artifacts; source files were made portable and private paths/PIDs removed during publication. `PUBLIC_MANIFEST.json` records the hashes of the actual public files and is the integrity reference for this repository.
