# 0.2.0 — Docker API / 容器接口服务

English / 中文

- Persistent CPU or Vulkan GPU inference, bounded asynchronous admission and warm native worker.
- Single-image embeddings, logical image batches, cosine similarity, health endpoints and OpenAPI.
- Upload limits, optional Bearer authentication, cancellation-safe IPC and explicit GPU failure without CPU fallback.
- Source Dockerfile, CPU/Adreno/DRM Compose examples, English and Simplified Chinese guides.
- Actual ARM64 Docker / Adreno 640 acceptance: GPU sequential HTTP mean 159.49 ms; batch4 throughput 6.42 images/s; four-image FP32-reference mean cosine 0.999951677.

- 常驻 CPU/Vulkan 特征服务，提供批量、相似度、健康检查及交互文档。
- 有界队列、超时、鉴权、上传限制；GPU 不可用时失败，不静默回退 CPU。
- 已在 855 的真实 ARM64 Docker 容器中验证硬件加速；提供中英文说明。

## Prebuilt image / 预编译镜像

The attached image is **Linux ARM64 only**, supports CPU and compatible Vulkan/Turnip at runtime, and contains **no model weights, game images, vendor drivers or API key**. The Adreno path requires the existing native Linux KGSL/Turnip host installation; it is not an Android APK or an NPU image.

附件镜像仅适用于 **Linux ARM64**。CPU/GPU 共用镜像，驱动与模型需自行只读挂载。不是 Android APK，不含 NPU 后端。

```bash
sha256sum -c dinov3-api-0.2.0-linux-arm64.tar.gz.sha256
gzip -dc dinov3-api-0.2.0-linux-arm64.tar.gz | docker load
docker tag dinov3-api:0.2.0-arm64 dinov3-api:local
# With your authorized model in ./gpu_conv_native:
docker compose up -d --no-build
```

For Adreno, use the additional `compose.adreno.yaml` and the actual device group ID. See `docs/API.md` / `docs/API.zh-CN.md` and `docs/SERVICE_VALIDATION.md` for paths, authentication, limits and measurement scope.

The prebuilt image was qualified using the `service-prebuilt` target and an open-source cross-compiled runner; the complete source-build Docker target is a separate CI check. No model/data access rights are granted by downloading this container.
