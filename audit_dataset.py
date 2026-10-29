"""Verify actual selected inputs, split disjointness and source-video isolation."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',type=Path);args=p.parse_args()
    rows=[json.loads(s) for s in (args.dataset/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    groups={}
    for game in dict.fromkeys(r['game'] for r in rows):
        records=[r for r in rows if r['game']==game]
        assert Counter(r['split'] for r in records)=={'calibration':24,'test':8}
        assert len(set(r['source'] for r in records))==32
        assert len(set(r['source_family'] for r in records))==32
        values=[int(r['dhash64'],16) for r in records]
        assert all((v^w).bit_count()>5 for i,v in enumerate(values) for w in values[:i])
        videos={s:{r['source_video_id'] for r in records if r['split']==s and r.get('source_video_id')} for s in ['calibration','test']}
        assert not videos['calibration']&videos['test']
        for r in records:
            raw=np.fromfile(args.dataset/r['raw_file'],dtype=np.float32)
            assert raw.size==np.prod(r['shape']) and np.isfinite(raw).all()
            assert Path(r['source']).is_file()
        groups[game]={'calibration':24,'test':8,'known_calibration_videos':sorted(videos['calibration']),
                      'known_test_videos':sorted(videos['test']),
                      'samples_without_video_id':sum(not r.get('source_video_id') for r in records)}
    assert len(groups)==6 and len(rows)==192
    result={'passed':True,'calibration':144,'evaluation':48,'groups':groups,
            'manifest_sha256':hashlib.sha256((args.dataset/'manifest.jsonl').read_bytes()).hexdigest(),
            'limitation':'No source-video independence can be proven for entries lacking video IDs; no object-detection accuracy is inferred.'}
    (args.dataset/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
