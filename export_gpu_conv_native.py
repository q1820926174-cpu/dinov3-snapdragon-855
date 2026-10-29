"""GPU-specific ConvNeXt: 1x1 Conv MLP with native LayerNorm and GELU."""
from pathlib import Path
from collections import Counter
import argparse, json, shutil, subprocess, sys
import torch
from torch import nn
from transformers import AutoModel
from export_snpe_compatible import pointwise
from export_onnx import FeatureModel


class ChannelNorm(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.norm = nn.LayerNorm(source.weight.numel(), eps=source.eps)
        self.norm.weight = nn.Parameter(source.weight.detach().flatten().clone())
        self.norm.bias = nn.Parameter(source.bias.detach().flatten().clone())

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.depthwise = source.depthwise_conv
        self.norm = ChannelNorm(source.layer_norm)
        self.expand = pointwise(source.pointwise_conv1)
        self.gelu = nn.GELU()
        self.project = pointwise(source.pointwise_conv2, source.gamma)

    def forward(self, x):
        return x + self.project(self.gelu(self.expand(self.norm(self.depthwise(x)))))


class Model(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.stages = nn.ModuleList()
        for stage in source.stages:
            down = [ChannelNorm(layer) if isinstance(layer, nn.LayerNorm) else layer
                    for layer in stage.downsample_layers]
            self.stages.append(nn.Sequential(*down, *[Block(layer) for layer in stage.layers]))
        self.norm = ChannelNorm(source.layer_norm)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.norm(x), self.norm(x.mean((2, 3), keepdim=True)).flatten(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('gpu_conv_native'))
    p.add_argument('--model', default='models/dinov3-convnext-tiny')
    p.add_argument('--pnnx', default=None, help='Optional PNNX executable path')
    a = p.parse_args()
    torch.set_num_threads(4)
    root = a.out; root.mkdir(parents=True,exist_ok=True)
    source = AutoModel.from_pretrained(a.model).eval()
    model = Model(source).eval()
    torch.manual_seed(42)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        expected = FeatureModel(source)(x)
        actual = model(x)
        errors = [float((v-r).abs().max()) for v,r in zip(actual,expected)]
        for v,r in zip(actual,expected):
            torch.testing.assert_close(v,r,rtol=1e-3,atol=1e-4)
        trace = torch.jit.trace(model, x, check_trace=False, strict=False)
        for v,r in zip(trace(x),actual):torch.testing.assert_close(v,r)
        trace.save(str(root/'model.pt'))
    (root/'pytorch_parity.json').write_text(json.dumps({'max_abs_dense_global':errors})+'\n')
    cli=a.pnnx or shutil.which('pnnx') or str(Path(sys.executable).parent/('pnnx.exe' if sys.platform=='win32' else 'pnnx'))
    with (root/'convert.log').open('wb') as f:
        subprocess.run([str(Path(cli).resolve()),
                        'model.pt','inputshape=[1,3,224,224]','fp16=0'],
                       cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    lines=(root/'model.ncnn.param').read_text().splitlines()
    mapping={'in0':'pixel_values','out0':'dense_features','out1':'global_embedding'}
    for i in range(2,len(lines)):
        lines[i]=' '.join(mapping.get(t,t) for t in lines[i].split())
    counts=Counter(line.split()[0] for line in lines[2:])
    assert counts['Gemm']==0, counts
    (root/'model.param').write_text('\n'.join(lines)+'\n')
    shutil.copyfile(root/'model.ncnn.bin',root/'model.bin')
    (root/'operators.json').write_text(json.dumps(counts,indent=2)+'\n')
    print('FP32 parity errors',errors,'operators',dict(counts),flush=True)


if __name__=='__main__':main()
