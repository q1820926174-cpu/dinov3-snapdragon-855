"""Read process-specific DRM execution counters around real HTTP requests.

Run on the Docker host with permission to inspect the native child's fdinfo.
Does not create/delete containers, change clocks or print API credentials.
"""
import argparse,json,os,re,subprocess,time,urllib.request
from pathlib import Path


def docker(*args):
    return subprocess.check_output(['docker',*args],text=True,timeout=20).strip()


def native_pid(container):
    for line in docker('top',container,'-eo','pid,comm').splitlines()[1:]:
        pid=int(line.split()[0])
        try:
            if Path(os.readlink(f'/proc/{pid}/exe')).name=='dinov3_embed_server':return pid
        except OSError:pass
    raise RuntimeError('native child not found or /proc access denied')


def snapshot(pid):
    clients={}
    for entry in Path(f'/proc/{pid}/fdinfo').iterdir():
        try:
            target=os.readlink(f'/proc/{pid}/fd/{entry.name}')
            if '/dev/dri/' not in target:continue
            fields={}
            for line in entry.read_text().splitlines():
                if line.startswith('drm-'):
                    name,value=line.split(':',1);fields[name]=value.strip()
            if fields:clients[fields.get('drm-client-id',entry.name)]={'device':target,'fields':fields}
        except OSError:continue
    return clients


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--container',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--requests',type=int,default=16)
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if not 1<=args.requests<=1000:p.error('requests must be 1..1000')
    key=os.getenv('DINO_API_KEY','')
    data=args.image.read_bytes();boundary='DinoDrmEvidenceBoundary'
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="probe.jpg"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+data+f'\r\n--{boundary}--\r\n'.encode())
    headers={'Content-Type':'multipart/form-data; boundary='+boundary}
    if key:headers['Authorization']='Bearer '+key
    def infer():
        req=urllib.request.Request(args.url.rstrip('/')+'/v1/embeddings',data=body,headers=headers)
        with urllib.request.urlopen(req,timeout=args.timeout) as response:result=json.load(response)
        if result['backend']!='gpu16s':raise RuntimeError('the selected API backend is not gpu16s')
        if len(result['data'][0]['embedding'])!=768:raise RuntimeError('invalid embedding response')
    for _ in range(3):infer()
    pid=native_pid(args.container);before=snapshot(pid)
    start=time.monotonic()
    for _ in range(args.requests):infer()
    elapsed=time.monotonic()-start;after=snapshot(pid);deltas={}
    for client,row in after.items():
        previous=before.get(client,{}).get('fields',{})
        for name,value in row['fields'].items():
            if 'engine' not in name and 'cycles' not in name:continue
            a=re.match(r'(\d+)',previous.get(name,''));b=re.match(r'(\d+)',value)
            if a and b:deltas[client+':'+name]={'before':previous[name],'after':value,'delta':int(b[1])-int(a[1])}
    proven=any(v['delta']>0 for v in deltas.values())
    result={'backend':'gpu16s','requests':args.requests,'elapsed_s':elapsed,'before':before,'after':after,
            'counter_deltas':deltas,'positive_execution_counter':proven,
            'scope':'Actual API native-child DRM counters; client IDs deduplicated. Not whole-GPU occupancy or a throughput benchmark.'}
    text=json.dumps(result,indent=2);print(text)
    if args.out:args.out.write_text(text+'\n')
    if not proven:raise SystemExit(2) # Unsupported counters or no positive evidence; do not invent a result.


if __name__=='__main__':main()
