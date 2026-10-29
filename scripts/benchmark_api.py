"""Benchmark a real HTTP endpoint; never prints credentials or image contents."""
import argparse,asyncio,json,os,time
from pathlib import Path
import httpx
import numpy as np


async def run(args):
    headers={}
    if os.getenv('DINO_API_KEY'):headers['Authorization']='Bearer '+os.environ['DINO_API_KEY']
    images=[(p.name,p.read_bytes()) for p in args.images]
    limit=asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url,headers=headers,timeout=args.timeout) as client:
        ready=await client.get('/health/ready');ready.raise_for_status()
        async def request(i):
            name,data=images[i%len(images)]
            async with limit:
                start=time.perf_counter()
                response=await client.post('/v1/embeddings',files={'file':(name,data)})
                response.raise_for_status();body=response.json()
                assert body['dimensions']==768 and body['normalized'] is True
                return {'wall_ms':(time.perf_counter()-start)*1000,'native_ms':body['data'][0]['native_ms'],
                        'queue_ms':body['timing']['queue_ms']}
        await request(0)
        start=time.perf_counter();records=await asyncio.gather(*(request(i) for i in range(args.requests)))
        elapsed=time.perf_counter()-start
    result={'backend':ready.json()['backend'],'requests':args.requests,'concurrency':args.concurrency,
        'wall_seconds':elapsed,'requests_per_second':args.requests/elapsed,
        'latency_mean_ms':float(np.mean([r['wall_ms'] for r in records])),
        'latency_p95_ms':float(np.percentile([r['wall_ms'] for r in records],95)),
        'native_mean_ms':float(np.mean([r['native_ms'] for r in records])),
        'queue_mean_ms':float(np.mean([r['queue_ms'] for r in records]))}
    print(json.dumps(result,indent=2))
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--images',type=Path,nargs='+',required=True)
    p.add_argument('--requests',type=int,default=16)
    p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--timeout',type=float,default=60)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if min(args.requests,args.concurrency,args.timeout)<=0:p.error('limits must be positive')
    asyncio.run(run(args))

if __name__=='__main__':main()
