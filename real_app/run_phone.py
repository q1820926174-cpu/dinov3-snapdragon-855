"""Run the unchanged upstream OmniMatch FastAPI /detect on the actual phone.

Only ProductionEmbedder is injected with a persistent local ncnn adapter.
There is no PyTorch, desktop inference, result replay or network inference.
"""
from pathlib import Path
import hashlib, importlib, io, json, os, struct, subprocess, sys, time, types
import numpy as np
from PIL import Image
import faiss

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
os.chdir(ROOT/'project')
sys.path.insert(0,str(ROOT/'project'))
faiss.omp_set_num_threads(1)
instances=[]
BACKEND='cpu'


class NcnnEmbedder:
    def __init__(self,model_name=None,cache_dir=None):
        self.backend=BACKEND
        prefix=BASE/('dinov3_conv_native' if BACKEND=='gpu16s' else 'dinov3_gpu_adaptive')
        self.log=(ROOT/(BACKEND+'_encoder.log')).open('ab')
        self.process=subprocess.Popen([str(Path(os.environ.get('DINO_EMBED_SERVER', str(BASE/'dinov3_embed_server')))),str(prefix),BACKEND],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        if self.read_exact(4)!=b'RDY1':raise RuntimeError('encoder not ready')
        self.inference_times=[]
        first=Image.open(ROOT/'gallery/reference_0.png').convert('RGB')
        for _ in range(3):self.embed_batch([first])
        self.inference_times=[]
        instances.append(self)

    def read_exact(self,n):
        data=bytearray()
        while len(data)<n:
            part=self.process.stdout.read(n-len(data))
            if not part:raise RuntimeError('encoder exited: '+str(self.process.poll()))
            data.extend(part)
        return bytes(data)

    def embed_batch(self,images):
        result=[]
        for image in images:
            image=image.convert('RGB')
            assert image.size==(224,224),'upstream crop/index must provide 224 RGB'
            self.process.stdin.write(image.tobytes());self.process.stdin.flush()
            data=self.read_exact(768*4+8)
            vector=np.frombuffer(data[:768*4],dtype='<f4').copy()
            self.inference_times.append(struct.unpack('<d',data[768*4:])[0])
            assert np.isfinite(vector).all()
            vector/=max(float(np.linalg.norm(vector)),1e-12)
            result.append(vector)
        return np.asarray(result,np.float32)

    def close(self):
        self.process.stdin.close()
        code=self.process.wait(timeout=15)
        self.process.stdout.close();self.log.close()
        if code:raise RuntimeError('encoder failed '+str(code))


# Keep upstream pipeline, FAISS index builder, FastAPI endpoint and NMS intact.
replacement=types.ModuleType('src.embedder')
replacement.ProductionEmbedder=NcnnEmbedder
sys.modules['src.embedder']=replacement
from src.build_index import build_production_index
from fastapi.testclient import TestClient
from app.api import app

manifest=json.loads((ROOT/'manifest.json').read_text())
results=[]
try:
    build_production_index('app/config.yaml')
    # Freeze the same CPU-reference gallery index for both execution backends.
    instances[-1].close()
    index_hash=hashlib.sha256((ROOT/'index/gallery.index').read_bytes()).hexdigest()
    for backend in ['gpu16s','cpu']:
        BACKEND=backend
        with TestClient(app) as client:
            encoder=app.state.pipeline.embedder
            for case in manifest['cases']:
                data=(ROOT/case['file']).read_bytes()
                assert hashlib.sha256(data).hexdigest()==case['sha256']
                encoder.inference_times=[]
                start=time.perf_counter()
                response=client.post('/detect',files={'file':(Path(case['file']).name,data,'image/jpeg')})
                elapsed=(time.perf_counter()-start)*1000
                assert response.status_code==200,(response.status_code,response.text)
                payload=response.json()
                record={'backend':backend,'case_id':case['id'],'http_status':response.status_code,
                        'api_wall_ms':elapsed,'windows':len(encoder.inference_times),
                        'encoder_total_ms':sum(encoder.inference_times),
                        'encoder_mean_ms':float(np.mean(encoder.inference_times)),
                        'response':payload}
                results.append(record)
                (ROOT/'api_results.json').write_text(json.dumps(results,indent=2)+'\n')
                print('APP_CASE',backend,case['id'],'windows',record['windows'],
                      'api_ms',round(elapsed,2),'detections',payload['count'],flush=True)
        encoder.close()
    proof={'project_commit':manifest['commit'],'fixed_cpu_index_sha256':index_hash,
           'pipeline_sha256':hashlib.sha256(Path('app/pipeline.py').read_bytes()).hexdigest(),
           'api_sha256':hashlib.sha256(Path('app/api.py').read_bytes()).hexdigest(),
           'model_sha256':{n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in
                          ['dinov3_embed_server','dinov3_conv_native.param','dinov3_conv_native.bin',
                           'dinov3_gpu_adaptive.param','dinov3_gpu_adaptive.bin']},
           'numpy':np.__version__,'faiss':faiss.__version__,
           'cpu_fallback':False,'all_compute_on_phone':True,
           'measurement':'TestClient in-process HTTP request through actual upstream route, JPEG decode, sliding windows, PIL resize, local IPC, ncnn normalization/inference, FAISS search, NMS and JSON; excludes model/index startup and file read before request. No GUI or live capture tested.'}
    (ROOT/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('REAL_APP_API_COMPLETE',flush=True)
finally:
    for encoder in instances:
        if encoder.process.poll() is None:
            try:encoder.close()
            except Exception:encoder.process.kill();encoder.process.wait()
