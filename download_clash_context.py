"""Sample real gameplay context crops across eight independent CR recordings.

Context crops supplement calibration; GamePhysics full frames remain held out.
These are not claimed to be full screenshots or ground-truth box annotations.
"""
import hashlib
import json
import tarfile
from pathlib import Path
from download_hf_samples import get

REPO='SudhishNeel/clash-royale-detection'

def main():
    listing=json.loads(get(f'https://huggingface.co/api/datasets/{REPO}/tree/main/bundles?limit=1000'))
    bundles=sorted((r for r in listing if r['path'].endswith('.tar')),key=lambda r:r['size'])[:8]
    records=[];root=Path('datasets_raw/clash_royale')
    for bundle in bundles:
        url=f'https://huggingface.co/datasets/{REPO}/resolve/main/'+bundle['path']
        archive=Path('downloads')/('clash_'+Path(bundle['path']).name)
        if not archive.exists(): archive.write_bytes(get(url))
        with tarfile.open(archive) as tar:
            members=[m for m in tar.getmembers() if '/ctx/' in m.name and m.name.endswith('.jpg')]
            members.sort(key=lambda m:int(Path(m.name).stem))
            count=min(64,len(members))
            for i in range(count):
                m=members[i*len(members)//count]
                video=m.name.split('/')[0]
                out=root/'train'/(video+'_ctx'+Path(m.name).name)
                out.parent.mkdir(parents=True,exist_ok=True)
                if not out.exists(): out.write_bytes(tar.extractfile(m).read())
                records.append({'file':out.relative_to(root).as_posix(),'dataset':REPO,'url':url,
                                'source_video_id':video,'source_member':m.name,'type':'real gameplay context crop',
                                'sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
        print(bundle['path'],'contexts',count,flush=True)
    # Keep provenance for the original supplementary bundle, too.
    target=root/'source_manifest_context.json'
    old=json.loads(target.read_text()) if target.exists() else []
    records=list({r['file']:r for r in old+records}.values())
    target.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__': main()
