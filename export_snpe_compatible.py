"""NCHW ConvNeXt lowering for legacy SNPE; GELU tanh approximation is explicit.

Official weights are retained. Verify outputs against the official FP32 model.
No edits to the installed SNPE SDK are required.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class Norm(nn.Module):
    scale = 1.0
    def __init__(self, source):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().reshape(1, -1, 1, 1))
        self.bias = nn.Parameter(source.bias.detach().reshape(1, -1, 1, 1))
        self.eps = source.eps

    def forward(self, x):
        x = x * self.scale
        centered = x - x.mean(1, keepdim=True)
        variance = (centered * centered).mean(1, keepdim=True)
        return centered * (variance + self.eps*self.scale*self.scale).pow(-0.5) * self.weight + self.bias


def pointwise(source, gamma=None):
    layer = nn.Conv2d(source.in_features, source.out_features, 1)
    weight = source.weight.detach().reshape(source.out_features, source.in_features, 1, 1)
    bias = source.bias.detach()
    if gamma is not None:
        weight = weight * gamma.detach().reshape(-1, 1, 1, 1)
        bias = bias * gamma.detach()
    layer.weight = nn.Parameter(weight.clone())
    layer.bias = nn.Parameter(bias.clone())
    return layer


class Block(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.depthwise = source.depthwise_conv
        self.norm = Norm(source.layer_norm)
        self.expand = pointwise(source.pointwise_conv1)
        self.project = pointwise(source.pointwise_conv2, source.gamma)

    def forward(self, x):
        z = self.expand(self.norm(self.depthwise(x)))
        # Hendrycks/Gimpel tanh approximation; replaces unsupported Erf/Gelu.
        z = 0.5 * z * (1 + torch.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)))
        return x + self.project(z)


class Compatible(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.stages = nn.ModuleList()
        for stage in source.stages:
            down = [Norm(layer) if isinstance(layer, nn.LayerNorm) else layer for layer in stage.downsample_layers]
            self.stages.append(nn.Sequential(*down, *[Block(layer) for layer in stage.layers]))
        self.norm = Norm(source.layer_norm)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.norm(x), self.norm(x.mean((2, 3), keepdim=True)).flatten(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/dinov3-convnext-tiny')
    p.add_argument('--out', default='dinov3_snpe_224.onnx')
    p.add_argument('--norm-scale',type=float,default=1.0,help='Exact LayerNorm rescaling to avoid half subnormal variance')
    p.add_argument('--norm-calibration',type=Path,help='Select per-LayerNorm exact power-of-two scales from calibration inputs only')
    args = p.parse_args()
    if args.norm_scale<=0: p.error('norm scale must be positive')
    Norm.scale=args.norm_scale
    torch.set_num_threads(4)
    source = AutoModel.from_pretrained(args.model).eval()
    lowered = Compatible(source).eval()
    if args.norm_calibration:
        from quantize_onnx_cpu import RawReader
        reader=RawReader(args.norm_calibration)
        stats={name:{'max_centered':0.0,'min_variance':float('inf')} for name,m in lowered.named_modules() if isinstance(m,Norm)}
        handles=[]
        def hook(name):
            def observe(module,inputs):
                z=inputs[0]-inputs[0].mean(1,keepdim=True)
                stats[name]['max_centered']=max(stats[name]['max_centered'],z.abs().max().item())
                stats[name]['min_variance']=min(stats[name]['min_variance'],(z*z).mean(1).min().item())
            return observe
        for name,m in lowered.named_modules():
            if isinstance(m,Norm):handles.append(m.register_forward_pre_hook(hook(name)))
        with torch.inference_mode():
            while (feed:=reader.get_next()) is not None:
                lowered(torch.from_numpy(feed['pixel_values']))
                if reader.index%24==0:print('norm calibration',reader.index,flush=True)
        for h in handles:h.remove()
        for name,m in lowered.named_modules():
            if isinstance(m,Norm):
                s=stats[name]
                m.scale=min(16.0,2.0**math.floor(math.log2(128/max(s['max_centered'],1e-6))))
                s['scale']=m.scale
                s['scaled_min_variance_plus_eps']=(s['min_variance']+m.eps)*m.scale*m.scale
                if s['scaled_min_variance_plus_eps']<2**-14:
                    print('WARNING half subnormal risk',name,s,flush=True)
        Path(args.out+'.norm_scales.json').write_text(json.dumps({'calibration_samples':reader.index,'layers':stats},indent=2)+'\n')
    torch.manual_seed(42)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        expected = source(x)
        dense, global_ = lowered(x)
        reference = expected.last_hidden_state[:, 1:, :].transpose(1, 2).reshape(1, 768, 7, 7)
        print('dense max abs', (reference-dense).abs().max().item(), flush=True)
        print('global cosine', F.cosine_similarity(global_, expected.pooler_output).item(), flush=True)
        torch.onnx.export(lowered, x, args.out, opset_version=13, dynamo=False,
                          input_names=['pixel_values'], output_names=['dense_features', 'global_embedding'])
    print('Exported', args.out, flush=True)


if __name__ == '__main__':
    main()
