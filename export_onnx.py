#!/usr/bin/env python3
"""CPU 导出 DINOv3 ConvNeXt-Tiny 的 768 维全局和稠密特征。

需先完成官方模型访问授权和 Hugging Face 登录。
SNPE 老版 ONNX 转换器可能不支持部分层；本脚本不能保证转换到 DLC。
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModel

MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"


class FeatureModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y = self.model(pixel_values=x)
        # HF DINOv3 ConvNeXt returns [pooled token, spatial tokens] in NLC.
        # The native final-stage stride is 32 for the supported input sizes.
        tokens = y.last_hidden_state[:, 1:, :].transpose(1, 2)
        dense = tokens.reshape(x.shape[0], -1, x.shape[2] // 32, x.shape[3] // 32)
        return dense, y.pooler_output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, choices=[224, 320, 448], default=224)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID or local model directory")
    p.add_argument("--out", type=Path, default=Path("dinov3_features_224.onnx"))
    args = p.parse_args()
    print("Loading official model on CPU; no CUDA required")
    model = AutoModel.from_pretrained(args.model).eval().cpu()
    wrapper = FeatureModel(model)
    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)
    with torch.inference_mode():
        dense, pool = wrapper(dummy)
        print("Dense:", tuple(dense.shape), "Global:", tuple(pool.shape))
        torch.onnx.export(
            wrapper, dummy, str(args.out),
            input_names=["pixel_values"],
            output_names=["dense_features", "global_embedding"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes=None,
        )
    print("Exported:", args.out)


if __name__ == "__main__":
    main()
