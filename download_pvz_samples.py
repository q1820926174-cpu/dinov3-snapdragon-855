"""Small real PvZ screenshot sample from a public YOLO project, with hashes."""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from download_hf_samples import get

REPO='pan0624/Yolov8-target-detection-and-simulation-click'

def main():
    tree=json.loads(get(f'https://api.github.com/repos/{REPO}/git/trees/HEAD?recursive=1'))
    commit=tree['sha']
    images=[r['path'] for r in tree['tree'] if r['path'].startswith('datasets/pvz/images/') and r['path'].endswith('.jpg')]
    records=[]
    # Preserve upstream train/val split hints, and sample across the full list.
    for split in ['train','val']:
        pool=sorted(p for p in images if f'/images/{split}/' in p)
        count=min(96 if split=='train' else 32,len(pool))
        for i in range(count):
            remote=pool[i*len(pool)//count]
            local=Path('datasets_raw/pvz/github')/split/'images'/Path(remote).name
            url=f'https://raw.githubusercontent.com/{REPO}/{commit}/{quote(remote)}'
            local.parent.mkdir(parents=True,exist_ok=True)
            if not local.exists(): local.write_bytes(get(url))
            label_remote=remote.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
            label=local.parent.parent/'labels'/(local.stem+'.txt')
            if any(r['path']==label_remote for r in tree['tree']):
                label.parent.mkdir(parents=True,exist_ok=True)
                if not label.exists(): label.write_bytes(get(f'https://raw.githubusercontent.com/{REPO}/{commit}/{quote(label_remote)}'))
            records.append({'file':local.relative_to('datasets_raw/pvz').as_posix(),'url':url,'repository':REPO,
                            'commit':commit,'upstream_split':split,'source_video_id':None,
                            'sha256':hashlib.sha256(local.read_bytes()).hexdigest()})
            print('pvz github',split,i+1,count,flush=True)
    Path('datasets_raw/pvz/source_manifest_github.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
