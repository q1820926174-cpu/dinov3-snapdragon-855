# Measurement scope and failures

## Model result

Six groups: PvZ, Hollow Knight, Terraria, Cuphead, Clash Royale and Genshin. Calibration had 144 images and engineering evaluation 48 images. Some sources have no source-video identity; disjoint filenames and near-duplicate screening do not prove independent videos. The 48 evaluation images were used during engineering optimization, not an unseen final validation.

The accepted graph has 188 nodes (40 Conv, 18 depthwise Conv, 24 LayerNorm, 18 GELU, no Gemm). Full feature output was compared to official FP32. Global mean cosine 0.999946, P05 0.999892, minimum 0.999860; dense mean 0.999899. Each of the six groups met the engineering global gate (mean ≥0.995 and P05 ≥0.98). These are not mAP or labeled matching recall.

Timing uses 12 fixed images, two/game, repeated three times, after three warmups. It excludes loading, warmup, file IO and NHWC input conversion, and includes extracting both outputs back to host memory. Dynamic clocks and 36 runs limit the precision of P95. No long thermal or power claim follows from this timing alone.

Failed controls are retained: full FP16 arithmetic averaged global cosine 0.967966 and failed; an all-native Gemm graph reached roughly 713 ms under its safe compatibility option, while its default driver path reported device lost. Fewer graph nodes alone did not make that graph faster. The old GPU FP32 depthwise path was particularly inefficient; this does not imply low GPU hardware peak FP32 throughput.

## Actual application

OmniMatch CV commit `bc410f1a410abc7d983278ddeb53b44850276b5d` provided the unmodified API, sliding window, FAISS and NMS implementation. Only the feature provider and local configuration were adapted. The original ViT backbone was replaced with this project's ConvNeXt-Tiny. Window size 64 and score threshold .78 were retained; stride changed from 16 to 64, batch size from 512 to 1. Last partial image columns were not added by the upstream scanner.

Three reference crops and four real 806×640 screenshots came from the pinned public PvZ dataset repository. Sources did not occur in the prior 192-image calibration/evaluation list; video independence is unknown. Three positive scenes contain four annotated class-0 targets; the fourth scene is negative.

The phone ran eight actual ASGI API requests, four/backend, 120 windows/request. GPU mean 14.794 s, CPU mean 26.749 s. All four CPU/GPU box sets were identical; maximum score difference .0021. This is backend acceptance using TestClient, not UI, TCP networking or live-game capture.

**Localization failed for both backends.** Sixteen predicted boxes, four ground-truth targets, no IoU≥.5 match. The 64×64 output window has a theoretical maximum IoU below .5 with each ~40×40 target, so zero recall must be interpreted with that geometric mismatch. Only two targets had a predicted-box center within their annotation; the negative frame returned four false detections. The application must be redesigned/calibrated before reliable use. Model cosine preservation is not application quality.

## Synchronized application energy

Two images/backend, counterbalanced order GPU0→CPU0→CPU1→GPU1. Each request processes 120 windows, with 8-second loaded-idle intervals before/after. Models are warmed before measurement. Four requests retained their earlier box outputs. All active energies are integrated from their own measurement interval; older timings were not multiplied by later wattage.

| Metric | GPU | CPU reference |
|---|---:|---:|
| Weighted mean system rail W | 3.6215 | 4.6423 |
| Mean request s | 14.8148 | 25.3984 |
| Gross J/screenshot | 53.6525 | 117.9072 |
| Approximate incremental J after idle subtraction | 40.1329 | 93.1167 |

GPU mean power was 22.0% lower and gross energy 54.5% lower in this short measurement. Individual GPU rounds were 3.493 / 3.750 W and CPU 4.690 / 4.597 W.

305 valid telemetry samples, median interval .5002 s, maximum .5460 s; VSYS refresh every five seconds, voltage range 4180–4359 mV. Trapezoid integration uses the same monotonic clock as the requests. Idle subtraction omits one second at both ends of each idle period and is approximate, not CPU/GPU die power.

Screen remained DPMS Off, USB connected and status Full. No charging, frequency, thermal or screen policy was changed. Battery activity temperatures were 37.1–38.1°C; local GPU sensor reached 72.9°C during GPU rounds, local CPU sensor 81.1°C during CPU rounds. Short dynamic-clock measurements are not long-run thermal qualification or calibrated external-meter measurements. Desktop/background services remained active. The known localization failures are unchanged.

## NPU limitations

Whole-model HTA compilation and whole-model W8A8 quality did not pass. A stem-only NPU/GPU prototype ran but represented ~0.318% of original MACs and failed the overall accuracy/speed goals. No high-NPU-occupancy deployment claim is made. GPU conversion and system-rail energy results must not be relabeled as NPU results.
