# Docker 与 API 部署

[English](API.md) | [简体中文](API.zh-CN.md)

## 服务能力与边界

每个容器常驻一个 ncnn 模型进程，启动时预热三次。HTTP 异步处理，推理使用有界 FIFO 队列；支持上传/像素限制、Bearer 鉴权、健康检查和 `/docs` 交互文档。超时请求不会打断本地管道读取，避免污染后续响应。

提供 **768 维、L2 归一化的全局图像特征**和两图余弦相似度。没有训练好的检测器、稠密特征接口，也不包含之前每图扫描 120 个窗口的 OmniMatch 业务流程。批量接口是多图请求分组，仍由单个常驻进程顺序推理，不宣称 GPU 张量批处理或多个模型实例并发。

支持 `cpu`（FP32，可配置线程数）及 `gpu16s`（Vulkan，FP16 packed 存储/FP32 运算）。缺少设备、软件 CPU Vulkan 驱动或不支持的层会明确失败，不自动回退 CPU。没有 CUDA、NPU 或全 FP16 运算后端。

## 从源码构建

先按 README 授权下载并转换模型。挂载目录必须包含 `model.param` 和 `model.bin`；镜像中不打包权重及厂商驱动。

```bash
docker build --target service -t dinov3-api:local .
export DINO_MODEL_DIR="$PWD/gpu_conv_native"
docker compose up -d
curl http://127.0.0.1:8000/health/ready
```

Dockerfile 在构建阶段编译固定版本、带补丁的 ncnn 和运行器，再把可执行文件复制到 Python 运行镜像。默认构建当前 Docker builder 的架构；在 x86 上交叉构建 ARM64 还需要已配置的 Buildx/QEMU，模拟构建不等于具备 Adreno 硬件加速。

已有可信、架构匹配的可执行文件时，可以快速打包：

```bash
mkdir -p .docker-build
cp build/dinov3_embed_server .docker-build/dinov3_embed_server
docker build --target service-prebuilt -t dinov3-api:local .
```

两种构建使用相同 API 和运行环境。手机 ARM64 容器验证使用预编译路径；运行器必须由本仓库固定版本和补丁编译，任意外部二进制不能证明禁止 CPU 回退。

## 骁龙 855 原生 Linux / Turnip

```bash
export DINO_GPU_GID=$(stat -c '%g' /dev/kgsl-3d0)
export DINO_MODEL_DIR="$PWD/gpu_conv_native"
docker compose -f compose.yaml -f compose.adreno.yaml up -d --build
```

此配置透传 KGSL 设备、加入对应数字设备组，并只读挂载已有 Turnip 驱动。例子使用 `/opt/scv41-mesa-kgsl` 及其中的 ARM64 ICD JSON，需与实际手机路径一致。容器以非 root、只读根文件系统运行，不需要 `--privileged`，也不能用 NVIDIA 的 `--gpus all` 替代 KGSL 透传。

`compose.vulkan.yaml` 是**尚未逐硬件验收的 DRM 示例**。配置 `DINO_GPU_GID`、`DINO_VK_DRIVER_DIR`、`DINO_VK_ICD`；ICD 路径须是容器内路径，JSON 内库路径及依赖也必须与镜像兼容。它不安装主机驱动，不保证任意 GPU 可用，也不是 Android APK。

**一加 6 / 骁龙 845 的 DRM 路径已单独测试**，使用 `Dockerfile.mesa` 和 `compose.mesa.yaml`。此层只在容器内安装开源 Mesa，避免主机驱动和 glibc 不匹配；`DINO_GPU_GID` 设置为 renderD128 的数字组号。详见 [845 报告与限制](SNAPDRAGON845.zh-CN.md)。当前 Mesa 22.3 兼容镜像的单图 GPU HTTP 略慢于 CPU，不应套用主机原生测速宣传。

## 调用示例

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
vector = response.json()["data"][0]["embedding"]  # 768 维归一化特征
```

`float` 是默认 JSON 数组编码；`base64` 是**小端 float32** 的 base64 编码。响应包含后端、维度、归一化标记、每图 `native_ms`，以及排队和处理时间。处理时间包含解码/缩放/本地调用；客户端耗时还包含上传、JSON 编码及传输。此处 native 时间只提取全局特征，不能与旧版两路输出 RAW runner 时间直接混用。

## 配置

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `DINO_BACKEND` | `cpu` | `cpu` 或 `gpu16s`，不自动回退 |
| `DINO_MODEL_PATH` | `/models/model` | 模型文件前缀 |
| `DINO_EXECUTABLE` | `/usr/local/bin/dinov3_embed_server` | 本地运行器 |
| `DINO_NUM_THREADS` | `4` | CPU 线程数，1–64 |
| `DINO_MAX_QUEUE` | `8` | 等待请求上限，不含正在执行的一个请求 |
| `DINO_TIMEOUT_SECONDS` | `30` | 请求截止时间/单次本地 IO 截止时间 |
| `DINO_STARTUP_TIMEOUT_SECONDS` | `60` | 本地进程就绪头等待时间 |
| `DINO_MAX_BODY_BYTES` | `8388608` | 整个请求体上限，包含批量 multipart 开销 |
| `DINO_MAX_BATCH` | `8` | 每批图像上限 |
| `DINO_MAX_IMAGE_PIXELS` | `20000000` | 源图像像素数上限 |
| `DINO_API_KEY` | 空 | 可选 Bearer 密钥，对外访问时应设置 |

Compose 默认仅监听 `127.0.0.1:8000`。用 `DINO_BIND_ADDRESS` 和 `DINO_PORT` 明确修改监听位置。跨网络部署应设置密钥并放在带 TLS 的反向代理后。密钥保存在本地环境或密钥管理器，不提交 Git、不打入镜像。健康接口不要求密钥，且不返回密钥或模型路径。

状态码：401 鉴权；413 请求过大；422 图像/批量参数错误；503 队列满或进程不可用（队列满带 `Retry-After: 1`）；504 请求超时。本地进程崩溃会使健康检查失败，请解决原因后重启容器；Docker 的 unhealthy 状态本身不会自动重启仍在运行的进程。

每个容器/GPU 使用**一个 Uvicorn worker**，避免重复加载权重和争用 GPU。镜像将 HTTP 并发连接限制为 32，超过上限也可能收到 Uvicorn 503。模型常驻，替换挂载文件后应重启，不要在服务运行时原地修改模型。

## 验证与测速

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/benchmark_api.py --url http://127.0.0.1:8000 --images a.jpg b.jpg --requests 16 --concurrency 4
```

测速工具从环境读取 `DINO_API_KEY`，报告真实 HTTP 延迟、吞吐及本地推理时间，不打印密钥。单元测试使用协议替身，不作为硬件加速证明；手机 ARM64 Docker/Turnip 验收另记在 `results/service/`。历史模型和滑窗应用测速仍见 `docs/MEASUREMENTS.md`，不能当成新 API 的端到端吞吐。

参考：[Docker 多阶段构建](https://docs.docker.com/get-started/docker-concepts/building-images/multi-stage-builds/)、[FastAPI 容器与 worker 内存](https://fastapi.tiangolo.com/deployment/docker/)。
