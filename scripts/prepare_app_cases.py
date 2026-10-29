"""Reconstruct measured application inputs from immutable public source files."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
APP_COMMIT='bc410f1a410abc7d983278ddeb53b44850276b5d'
DATA_COMMIT='a6e1d91bc4cb632351f1c35a8e104156cbc3de13'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',type=Path,help='Optional existing pvz/github directory')
    p.add_argument('--app-source',type=Path,help='Optional existing pinned OmniMatch checkout')
    a=p.parse_args()
    root=ROOT/'real_app';external=a.app_source or ROOT/'external/OmniMatch_CV'
    if not external.exists():
        external.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','https://github.com/drtiwari/OmniMatch_CV',str(external)],check=True)
        subprocess.run(['git','-C',str(external),'checkout','--detach',APP_COMMIT],check=True)
    head=subprocess.check_output(['git','-C',str(external),'rev-parse','HEAD'],text=True).strip()
    if head!=APP_COMMIT:raise RuntimeError('Upstream checkout must match '+APP_COMMIT)
    for name in ['app','src']:
        shutil.copytree(external/name,root/'project'/name,dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    manifest=json.loads((ROOT/'results/real_app/manifest.json').read_text(encoding='utf-8'))
    from PIL import Image
    for group in ['gallery','cases']:
        for record in manifest[group]:
            relative=record['source'].split('/github/',1)[1]
            split=relative.split('/')[0];name=Path(relative).name
            if a.source_root:
                data=(a.source_root/relative).read_bytes()
            else:
                url=f'https://raw.githubusercontent.com/pan0624/Yolov8-target-detection-and-simulation-click/{DATA_COMMIT}/datasets/pvz/images/{split}/{name}'
                with urlopen(url,timeout=45) as response:data=response.read()
            if hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise RuntimeError('Source hash mismatch: '+name)
            original=root/'source_images'/split/name;original.parent.mkdir(parents=True,exist_ok=True)
            original.write_bytes(data)
            target=root/record['file'];target.parent.mkdir(parents=True,exist_ok=True)
            if group=='gallery':
                Image.open(original).convert('RGB').crop(record['box']).resize((224,224),Image.Resampling.NEAREST).save(target)
            else:target.write_bytes(data)
    import yaml
    (root/'project/app/config.yaml').write_text(yaml.safe_dump(manifest['config']))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    shutil.copyfile(ROOT/'results/real_app/api_results.json',root/'api_results.json')
    print('Prepared three gallery crops, four cases and pinned upstream app; weights are not downloaded.')

if __name__=='__main__':main()
