"""Check the public allowlist boundaries without printing matching file contents."""
from pathlib import Path
import json,re,subprocess

ROOT=Path(__file__).resolve().parents[1]
try:
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--cached'],text=True).splitlines()
except subprocess.CalledProcessError:
    names=[]
if not names:
    names=[str(p.relative_to(ROOT)).replace('\\','/') for p in ROOT.rglob('*')
           if p.is_file() and not any(v in p.parts for v in ['.git','__pycache__','artifacts','build','external','project','gallery','cases','source_images'])]
forbidden={'.bin','.param','.pt','.pth','.onnx','.dlc','.safetensors','.raw','.npy','.npz','.jpg','.png','.jpeg','.mp4','.whl','.tar','.exe','.dll','.so','.key','.pem'}
patterns=[re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(r'hf_[A-Za-z0-9]{30,}'),
          re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),re.compile(r'[A-Za-z]:[\\/](?:Users|项目)[\\/]')]
errors=[]
for name in names:
    p=ROOT/name
    if any(part in ('.private','.hf-auth') for part in p.parts):errors.append((name,'private directory'))
    if p.suffix.lower() in forbidden:errors.append((name,'forbidden artifact type'))
    if p.stat().st_size>2_000_000:errors.append((name,'unexpected large file'))
    text=p.read_text(encoding='utf-8',errors='replace')
    for pattern in patterns:
        if pattern.search(text):errors.append((name,'credential/private-path pattern'))
    if p.suffix=='.json':json.loads(text)
    if p.suffix=='.jsonl':
        for line in text.splitlines():json.loads(line)
if errors:
    for name,reason in errors:print(name+': '+reason)
    raise SystemExit(1)
print('Public source/evidence boundary checks passed:',len(names),'files')
