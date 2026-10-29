# DINOv3 on Snapdragon 855 / Adreno 640

[English](README.md) | [简体中文](README.zh-CN.md)

## Docker 与硬件加速 API

新增常驻模型的 HTTP API：单图/多图特征、两图相似度、健康检查和 `/docs`。支持 CPU FP32 或 Vulkan GPU 混合精度；包括有界队列、超时、鉴权与上传限制。模型及驱动只读挂载，不进入镜像。

```bash
docker compose up -d --build  # 默认 CPU
# 855 原生 Linux + 已安装的 Turnip：
export DINO_GPU_GID=$(stat -c '%g' /dev/kgsl-3d0)
docker compose -f compose.yaml -f compose.adreno.yaml up -d --build
curl -F file=@image.jpg http://127.0.0.1:8000/v1/embeddings
```

先按下文生成 `gpu_conv_native/model.param` 和 `model.bin`。完整说明见 [中文 API 文档](docs/API.zh-CN.md) / [English API guide](docs/API.md)。API 是全局特征服务，不是已通过验收的目标检测器。

在三星 S10 SCV41 的原生 Linux + Mesa Turnip 环境上，将同一套 **DINOv3 ConvNeXt-Tiny** 权重转换为更适合 Adreno 640 的 ncnn Vulkan 图，并实测特征精度、真实应用后端速度及系统能耗。

This repository contains conversion code, native runners, the tested ncnn compatibility patch, and measurement records. It does **not** contain model weights, game images, Qualcomm SDKs, phone firmware, credentials, or a ready-made Android APK.

## 实测结果

### 单次模型推理，224×224

| 路径 | 平均时间 | 推理 FPS |
|---|---:|---:|
| 优化 GPU：原生 LayerNorm/GELU + 1×1 Conv MLP | **121.58 ms** | **8.23** |
| 旧 GPU：SNPE 兼容算子拆解图 | 226.40 ms | 4.42 |
| 旧图 CPU FP32，同期参考 | 185.05 ms | 5.40 |

相对旧 GPU，吞吐提高 **1.86×**、耗时下降 **46.3%**。同一官方模型，没有换小模型或重新训练。六游戏共 48 张工程验证图，全局特征平均余弦 **0.999946**、P05 **0.999892**，空间特征平均余弦 **0.999899**。它们是特征保持指标，不是检测 mAP；这些图曾用于工程调优，不是新的独立盲测。

### 真实应用后端与功耗

接入 [OmniMatch CV](https://github.com/drtiwari/OmniMatch_CV) 的原有 FastAPI `/detect`、滑窗、FAISS 检索和 NMS 流程。将其特征提取器换成这里的 ConvNeXt-Tiny，实际计算均在手机上完成。

| 同步功耗测试，两张完整截图/后端 | 优化 GPU | CPU FP32 参考图 |
|---|---:|---:|
| 平均系统供电轨功率 | **3.62 W** | 4.64 W |
| 完整截图处理时间 | **14.81 s** | 25.40 s |
| 每张截图系统能耗 | **53.65 J** | 117.91 J |

每图实际扫描 **120 个窗口**。GPU 的系统能耗低约 **54.5%**，但这不是实时完整截图检测。功率来自板载 MAX77705 只读遥测、屏幕关闭且后台服务保持运行，是系统 DC 供电轨估算，**不是芯片单独功耗或外接功率计读数**。两图/后端的短测不代表长期续航。

此前四图应用测试中，两种后端返回的框完全一致，置信度最大相差 0.0021。**应用定位验收未通过**：固定 64×64 输出框和约 40×40 标注目标不匹配，且负样本有误检。没有把接近 CPU 的结果表述成可用的检测准确率。此处测试的是实际后端 ASGI 路由，不包含 Streamlit 界面、TCP 网络和实时截图采集。

## 转换要点

- 保留原生 LayerNorm，通道归一化前后进行布局变换。
- 用 ncnn 原生 GELU 取代多个小算子调度；当前 Vulkan GELU 使用 tanh 近似。
- MLP 的 Linear 改写为 1×1 Conv，gamma 折入 project 权重/偏置，避开本机较慢的 Gemm 路径。
- 使用 PNNX。优化后 **188 节点、0 Gemm**；将所有原生 Linear 都保留为 Gemm 的对照图反而约 713 ms，未被选用。
- 推荐 `gpu16s`：**FP16 packed 存储 + FP32 运算**。开启 FP16 运算的对照精度未通过。
- 运行器检查 Vulkan 支持，配合 ncnn 补丁的 `SCV41_NCNN_STRICT_GPU=1` 禁止模型算子回退 CPU。应用预处理、检索与调度仍使用 CPU。

## 下载与导出模型

先自行在 [官方模型页面](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m) 获得访问权限并接受适用条款。权重及其衍生物不受本仓库 MIT 许可证覆盖，也不随仓库分发。

```bash
python -m venv .venv
# Activate the virtual environment for your shell, then:
python -m pip install -r requirements.txt
hf auth login
hf download facebook/dinov3-convnext-tiny-pretrain-lvd1689m --local-dir models/dinov3-convnext-tiny
python export_gpu_conv_native.py --model models/dinov3-convnext-tiny --out gpu_conv_native
```

得到 `gpu_conv_native/model.param` 和 `model.bin`。权重文件仍为 FP32 序列化，运行时才使用半精度 packed 存储。导出时会核对 PyTorch 重写图、官方 FP32 和 TorchScript 的数值一致性；手机精度仍需另行验证。

## 编译与设备运行

测试依赖 ncnn commit `9fc8106e6634a60f39945d63e5ed2dd0704ac9fc` 和本仓库的补丁。需要可用的 Vulkan/Turnip 驱动；本仓库不安装或刷写手机系统。见 [构建与复现说明](docs/REPRODUCE.md)。

```bash
git clone https://github.com/Tencent/ncnn.git third_party/ncnn
git -C third_party/ncnn checkout 9fc8106e6634a60f39945d63e5ed2dd0704ac9fc
git -C third_party/ncnn submodule update --init --recursive
git -C third_party/ncnn apply ../../patches/ncnn-scv41.patch
cmake -S third_party/ncnn -B build-ncnn-lib -DNCNN_VULKAN=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$PWD/ncnn-install"
cmake --build build-ncnn-lib -j4
cmake --install build-ncnn-lib
cmake -S . -B build -Dncnn_DIR="$PWD/ncnn-install/lib/cmake/ncnn"
cmake --build build -j4
```

以上命令是**本机架构构建**；在 x86 主机执行不会自动得到 ARM64 二进制。交叉编译需给两次 CMake 配置都提供 ARM64 toolchain/sysroot，见文档。

将模型和 ARM64 runner 放到手机后，在已配置 Vulkan ICD 的环境中执行：

```bash
./dinov3_ncnn model gpu16s inputs.txt output
```

`inputs.txt` 每行一个 224×224×3 **NHWC、float32、已归一化** RAW 的手机路径。预处理使用 letterbox 114 灰填充及 ImageNet mean/std，必须与参考一致。输出含全局向量、空间特征及计时 CSV；计时排除模型加载、预热、RAW 文件 IO 和输入布局转换。

`dinov3_embed_server` 是持久模型进程，接收固定 224 RGB 字节并输出全局特征，供 `real_app/` 适配器使用。它的二进制协议是本机小端布局，不能直接作为通用网络 API。

## 数据与报告

- [完整测量说明](docs/MEASUREMENTS.md)：数据范围、基线差异、功耗口径、失败结论。
- `results/`：聚合指标、实际 API 响应、功率原始时间序列和时钟标记。已去除本地绝对路径及屏幕进程信息，没有原图或特征二进制。
- `download_*`、`prepare_multigame_calibration.py`、`evaluate_features.py`：数据获取、预处理和特征保持检查。请遵守来源许可和访问条件。
- `scripts/prepare_app_cases.py`：从固定公开来源重建四图应用样例及参考模板，并获取固定版本的上游应用；不在 Git 中分发游戏素材。
- `real_app/power_monitor.py` / `power_worker.py`：只读板载采样和应用负载。需要相同硬件接口以及既有计算锁，不能直接套用于任意 Android 手机。
- `python scripts/summarize_power.py`：从公开原始遥测和阶段时间标记重新计算功率、积分能耗及空闲基线。

## NPU 状态

本仓库发布的是 **GPU 方案**。旧 SNPE 整图 HTA 编译、整网 INT8 精度未通过；曾跑通的小段 NPU → GPU 原型只覆盖原模型约 **0.318% MAC**，未满足精度/速度目标。没有声称高 NPU 占用、完整 NPU 主干或 NPU/GPU 协同加速成功。

## License

原创代码采用 [MIT](LICENSE)。第三方 ncnn 补丁上下文保留其 BSD-3-Clause 许可证，见 [第三方说明](THIRD_PARTY_NOTICES.md)。DINOv3 模型、上游应用、游戏数据和 Qualcomm SDK 分别受各自条款约束。
