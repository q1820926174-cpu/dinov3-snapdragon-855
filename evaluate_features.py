"""Audit same-input global and local feature preservation; no detection claim."""
import argparse
import json
from pathlib import Path
import numpy as np

def unit(x,axis):
    if not np.isfinite(x).all():raise ValueError('non-finite outputs')
    norm=np.linalg.norm(x,axis=axis,keepdims=True)
    if (norm<1e-12).any():raise ValueError('zero vectors')
    return x/norm

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    rows=[json.loads(s) for s in (args.reference/'feature_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    candidate_rows=[json.loads(s) for s in (args.candidate/'feature_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if rows!=candidate_rows:raise ValueError('output/input ordering mismatch')
    a=np.load(args.reference/'features_fp32.npy');b=np.load(args.candidate/'features_fp32.npy')
    da=np.load(args.reference/'features_fp32_dense.npy');db=np.load(args.candidate/'features_fp32_dense.npy')
    if a.shape!=b.shape or da.shape!=db.shape:raise ValueError('shape mismatch')
    cosine=(unit(a,1)*unit(b,1)).sum(1)
    ua=unit(da,1).reshape(len(a),768,49).transpose(0,2,1)
    ub=unit(db,1).reshape(len(a),768,49).transpose(0,2,1)
    local=(ua*ub).sum(2)
    position=(ub@ua.transpose(0,2,1)).argmax(2)==np.arange(49)[None,:]
    def stats(indices):
        g=cosine[indices];d=local[indices]
        return {'n':len(g),'global_mean_cosine':float(g.mean()),'global_p05_cosine':float(np.percentile(g,5)),
                'global_min_cosine':float(g.min()),'dense_mean_cosine':float(d.mean()),
                'dense_p05_cosine':float(np.percentile(d,5)),
                'same_image_patch_position_recall':float(position[indices].mean()),
                'passes_global_gate':bool(g.mean()>=.995 and np.percentile(g,5)>=.98)}
    result={'overall':stats(np.arange(len(a))),
            'games':{g:stats([i for i,r in enumerate(rows) if r['game']==g]) for g in dict.fromkeys(r['game'] for r in rows)},
            'per_sample':[{'game':r['game'],'source':r['source'],'global_cosine':float(cosine[i])} for i,r in enumerate(rows)],
            'notice':'Feature preservation only. Patch-position consistency is not labeled object matching Recall/mAP.'}
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'overall':result['overall'],'games':result['games']},indent=2),flush=True)

if __name__=='__main__':main()
