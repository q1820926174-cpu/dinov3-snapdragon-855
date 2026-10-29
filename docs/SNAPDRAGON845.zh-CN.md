# 一加 6 / 骁龙 845 验收

[English](SNAPDRAGON845.md) | [简体中文](SNAPDRAGON845.zh-CN.md)

实际连接一加 6，设备树确认 `oneplus,enchilada` / `qcom,sdm845`，GPU 为 Adreno 630。原生 Linux 7.3.0、Mesa 26.1.6、Vulkan loader 1.4.363，使用 `/dev/dri/renderD128`。保留桌面、Waydroid、SSH、Docker 等原有服务；没有改固件、内核、主机驱动、充电或频率策略。

## 原生模型

与 855 完全相同的已验收模型，224 输入、CPU 四线程、GPU 禁止模型算子回退 CPU。48 张同输入质量验证；测速仍为每游戏两张、共 12 张重复三次，预热三次后计时，提取全局与空间两路输出。

| 后端 | 平均 ms | P95 ms | 推理 FPS |
|---|---:|---:|---:|
| CPU FP32 | 206.07 | 210.21 | 4.85 |
| GPU 半精度 packed 存储/FP32 运算 | **147.27** | **150.68** | **6.79** |

GPU 对官方 FP32 的全局平均余弦 **0.999946**、P05 **0.999892**、最小 **0.999860**，空间平均余弦 **0.999883**，六游戏组都通过既定特征保持门槛。CPU 在 FP32 数值精度下近乎一致。这里不是目标检测准确率。

此前同图/同输入子集的 855 GPU 为 121.58 ms / 8.23 FPS，本次 845 延迟约高 21%。两台设备的内核、Mesa、内存、后台和温度不同，不能当作芯片峰值对比；旧文档 CPU 参考图也不同，不能直接把它当作同优化图的 855 CPU 对照。

## 真实 Docker API

845 主机 Mesa 使用的 glibc 比已发布 Bookworm 镜像更新。为避免拷贝不兼容的主机库，也不更换主机驱动，新增 `Dockerfile.mesa`，仅在容器中安装 Bookworm 的开源 Mesa 22.3.6。模型、运行器和 API 与公开 ARM64 基础镜像相同。原生进程日志确认实际使用 Turnip Adreno 630，而非软件 Vulkan。

| API 模式 | CPU | GPU |
|---|---:|---:|
| 串行 HTTP 平均 | **232.31 ms** | 246.01 ms |
| 串行 HTTP P95 | **242.64 ms** | 278.58 ms |
| 串行吞吐 | **4.30 图/秒** | 4.06 图/秒 |
| 四图批量吞吐 | 4.01 图/秒 | **4.29 图/秒** |

单图/每种并发各 16 次，循环四张真实截图；批量模式四次请求，共 16 图。客户端在隔离的 Python 3.11/HTTPX 容器中，通过 host 网络访问服务实际映射的 localhost 端口。两种后端加载同一个优化图；服务用户 10001、根目录只读、非 privileged，只透传 DRM 渲染节点并只读挂载模型。

四张 API 输出对官方 FP32：GPU 平均余弦 **0.999952**、最小 **0.999943**，CPU 接近 1。鉴权、错误图片、批量上限、相似度、单图/多图以及串行/四客户端调用都通过。

**当前兼容 Docker 栈的单图 GPU 延迟没有优于 CPU**；CPU 略快，四图批量则 GPU 略快。主机原生 GPU 明显快于原生 CPU。驱动版本与 API 路径不同，不能把所有差异归因于硬件或转换。如果使用这一个兼容镜像且优先单次延迟，建议按实测选择 CPU；不能把原生 6.79 FPS 标成 Docker HTTP 吞吐。

测试结束后已删除临时容器和测试密钥；没有改动 855 正在运行的 API，也没有在 845 上留下常驻测试服务。

## DRM 容器复现

先构建 API 基础镜像，或加载公开 v0.2.0 ARM64 镜像：

```bash
docker tag dinov3-api:0.2.0-arm64 dinov3-api:local
export DINO_GPU_GID=$(stat -c '%g' /dev/dri/renderD128)
export DINO_MODEL_DIR="$PWD/gpu_conv_native"
docker compose -f compose.yaml -f compose.mesa.yaml up -d --build
```

此层打包开源 Mesa，不挂载主机 Vulkan 库。默认 ICD 是 ARM64 Freedreno，其他架构/GPU 需正确设置 `DINO_VK_ICD`。主机仍需受支持的 DRM 设备和权限，不适用于原厂 Android，也不能代替 855 的 KGSL 配置；855 使用单独验收的 `compose.adreno.yaml`。

## 功耗边界与证据

本轮没有获得与 855 相同的 ISYS×VSYS 系统供电轨通道。一加提供的电池和 USB 充电遥测不能在未核对充放电流向、损耗前，直接代入 855 的系统功耗口径。因此不声称 845 的瓦数、焦耳或续航结果。

`results/snapdragon845/native.json`、原生计时 CSV、`docker-mesa22.json` 保留实际结果、软件版本、模型及输入哈希。均为短时动态频率测试；没有训练模型，没有实时游戏自动操作或新的定位质量结论。
