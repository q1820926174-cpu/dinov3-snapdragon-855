"""Counterbalanced full-app power workload, with the same local ncnn adapter."""
from pathlib import Path
import hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parent
os.environ.update(LD_LIBRARY_PATH='/opt/scv41-mesa-kgsl/lib',XDG_RUNTIME_DIR='/run/user/1000',
    VK_DRIVER_FILES='/opt/scv41-mesa-kgsl/share/vulkan/icd.d/freedreno_icd.armv8-a.json',
    SCV41_NCNN_STRICT_GPU='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4')
sys.path.insert(0,str(ROOT))
import power_app_adapter as adapter
from fastapi.testclient import TestClient
from app.api import app

OUT=ROOT/'power';OUT.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'manifest.json').read_text())
previous=json.loads((ROOT/'api_results.json').read_text())
events=(OUT/'events.jsonl').open('w',buffering=1)
def event(kind,**fields):
    row={'event':kind,'time_s':time.monotonic(),**fields}
    events.write(json.dumps(row)+'\n')

jobs=[('gpu16s',0),('cpu',0),('cpu',1),('gpu16s',1)]
results=[]
event('worker_start',order=jobs)
try:
    for job,(backend,case_id) in enumerate(jobs):
        adapter.BACKEND=backend
        with TestClient(app) as client:
            encoder=app.state.pipeline.embedder
            event('idle_before_start',job=job,backend=backend,case_id=case_id)
            time.sleep(8)
            event('idle_before_end',job=job)
            encoder.inference_times=[]
            case=manifest['cases'][case_id]
            data=(ROOT/case['file']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==case['sha256']
            event('request_start',job=job,backend=backend,case_id=case_id)
            start=time.monotonic()
            response=client.post('/detect',files={'file':(Path(case['file']).name,data,'image/jpeg')})
            end=time.monotonic()
            event('request_end',job=job)
            assert response.status_code==200,(response.status_code,response.text)
            payload=response.json()
            expected=next(r['response'] for r in previous if r['backend']==backend and r['case_id']==case_id)
            boxset=lambda r:{tuple(d['box'][k] for k in ['x1','y1','x2','y2']) for d in r['detections']}
            assert boxset(payload)==boxset(expected),'application output changed'
            assert len(encoder.inference_times)==120
            event('idle_after_start',job=job)
            time.sleep(8)
            event('idle_after_end',job=job)
            results.append({'job':job,'backend':backend,'case_id':case_id,'start_s':start,'end_s':end,
                'api_wall_ms':(end-start)*1000,'windows':len(encoder.inference_times),
                'encoder_ms':sum(encoder.inference_times),'response':payload})
            (OUT/'workload_results.json').write_text(json.dumps(results,indent=2)+'\n')
            print('POWER_JOB',job,backend,case_id,'api_s',round(end-start,3),flush=True)
        encoder.close()
    event('worker_end')
finally:
    for encoder in adapter.instances:
        if encoder.process.poll() is None:
            try:encoder.close()
            except Exception:encoder.process.kill();encoder.process.wait()
    events.close()
