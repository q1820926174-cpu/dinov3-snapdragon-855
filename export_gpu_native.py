"""GPU-only export preserving native LayerNorm, Linear, GELU and original weights.

No legacy SNPE operator decomposition, gamma folding or range scaling is applied.
PNNX is used as recommended by ncnn. Device quality must be verified separately.
"""
from pathlib import Path
import subprocess,sys,json
import torch
from transformers import AutoModel
from export_onnx import FeatureModel

torch.set_num_threads(4)
root=Path('gpu_native');root.mkdir(exist_ok=True)
model=AutoModel.from_pretrained('models/dinov3-convnext-tiny').eval()
wrapper=FeatureModel(model).eval()
torch.manual_seed(42);x=torch.randn(1,3,224,224)
with torch.inference_mode():
    expected=wrapper(x)
    trace=torch.jit.trace(wrapper,x,check_trace=False,strict=False)
    actual=trace(x)
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)
    trace.save(str(root/'dinov3.pt'))
print('Native TorchScript parity passed; converting with PNNX',flush=True)
import shutil
cli=Path(shutil.which('pnnx') or (Path(sys.executable).parent/('pnnx.exe' if sys.platform=='win32' else 'pnnx')))
with (root/'convert.log').open('wb') as log:
    subprocess.run([str(cli.resolve()),'dinov3.pt','inputshape=[1,3,224,224]','fp16=0'],cwd=root,
                   stdout=log,stderr=subprocess.STDOUT,check=True)
p=root/'dinov3.ncnn.param'
lines=p.read_text().splitlines()
mapping={'in0':'pixel_values','out0':'dense_features','out1':'global_embedding'}
for i,line in enumerate(lines):
    if i<2:continue
    tokens=[mapping.get(token,token) for token in line.split()]
    if tokens[0]=='Squeeze':
        assert tokens[1]=='squeeze_128' and tokens[5]=='global_embedding'
        tokens=['Reshape']+tokens[1:6]+['0=768']
    lines[i]=' '.join(tokens)
(root/'model.param').write_text('\n'.join(lines)+'\n')
import shutil
shutil.copyfile(root/'dinov3.ncnn.bin',root/'model.bin')
from collections import Counter
counts=Counter(line.split()[0] for line in lines[2:])
(root/'operators.json').write_text(json.dumps(counts,indent=2)+'\n')
print('Converted GPU-specific operators:',dict(counts),flush=True)
