"""HTTP API; OpenAPI descriptions include English and Chinese."""
import asyncio
import base64
import hmac
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import Depends,FastAPI,File,HTTPException,Request,UploadFile
from fastapi.responses import JSONResponse,ORJSONResponse
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
from .engine import BusyError,Engine,InvalidImageError,Settings,UnavailableError


class BodyLimit:
    def __init__(self,app,limit):self.app=app;self.limit=limit
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope['method'] not in ('POST','PUT','PATCH'):
            return await self.app(scope,receive,send)
        headers=dict(scope['headers'])
        try:length=int(headers.get(b'content-length',b'0'))
        except ValueError:length=self.limit+1
        if length>self.limit:
            return await JSONResponse({'detail':'request body too large'},status_code=413)(scope,receive,send)
        messages=[];total=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            total+=len(message.get('body',b''))
            if total>self.limit:
                return await JSONResponse({'detail':'request body too large'},status_code=413)(scope,receive,send)
            messages.append(message)
            if not message.get('more_body',False):break
        async def replay():
            if messages:return messages.pop(0)
            return await receive()
        await self.app(scope,replay,send)


def create_app(settings=None,engine_factory=Engine):
    cfg=settings or Settings.from_env()
    @asynccontextmanager
    async def lifespan(app):
        engine=engine_factory(cfg);app.state.engine=engine
        await engine.start()
        try:yield
        finally:await engine.close()
    app=FastAPI(title='DINOv3 ncnn API / 特征服务',version='0.2.0',lifespan=lifespan,default_response_class=ORJSONResponse,
        description='Persistent CPU or Vulkan embedding service. 常驻 CPU/Vulkan 特征服务；不提供目标检测。')
    app.add_middleware(BodyLimit,limit=cfg.max_body_bytes)
    bearer=HTTPBearer(auto_error=False)
    async def authenticate(credentials: HTTPAuthorizationCredentials|None=Depends(bearer)):
        if cfg.api_key:
            supplied=credentials.credentials if credentials and credentials.scheme.lower()=='bearer' else ''
            if not hmac.compare_digest(supplied.encode('utf-8'),cfg.api_key.encode('utf-8')):
                raise HTTPException(401,'invalid API key',headers={'WWW-Authenticate':'Bearer'})

    def status(request):
        engine=request.app.state.engine
        return {'ready':engine.ready,'backend':cfg.backend,'runtime':'ncnn-vulkan' if cfg.backend=='gpu16s' else 'ncnn-cpu',
                'allow_cpu_fallback':False,'queue_depth':engine.queue.qsize(),'queue_capacity':cfg.queue_size,
                'active':engine.active,'completed_images':engine.completed_images,'dimensions':768}

    @app.get('/health/live',summary='Liveness / 存活状态')
    async def live(request: Request):
        result=status(request)
        return JSONResponse(result,status_code=200 if result['ready'] else 503)

    @app.get('/health/ready',summary='Model readiness / 模型就绪状态')
    async def ready(request: Request):return await live(request)

    @app.get('/v1/models',dependencies=[Depends(authenticate)],summary='Model metadata / 模型信息')
    async def models(request: Request):
        return {'model':'dinov3-convnext-tiny','input_size':[224,224],
                'preprocess':'EXIF transpose, RGB, bicubic letterbox 114, ImageNet normalization',**status(request)}

    async def infer(request,files):
        if not 1<=len(files)<=cfg.max_batch:raise HTTPException(422,'batch size out of range')
        images=[]
        for file in files:
            try:data=await file.read(cfg.max_body_bytes+1)
            finally:await file.close()
            if not data:raise HTTPException(422,'empty image')
            if len(data)>cfg.max_body_bytes:raise HTTPException(413,'image too large')
            images.append(data)
        try:return await request.app.state.engine.infer(images)
        except BusyError as error:raise HTTPException(503,str(error),headers={'Retry-After':'1'}) from error
        except UnavailableError as error:raise HTTPException(503,str(error)) from error
        except InvalidImageError as error:raise HTTPException(422,str(error)) from error
        except asyncio.TimeoutError as error:raise HTTPException(504,'inference deadline exceeded') from error

    def encode(result,encoding):
        data=[]
        for i,(vector,ms) in enumerate(zip(result['vectors'],result['native_ms'])):
            value=vector.tolist() if encoding=='float' else base64.b64encode(vector.astype('<f4').tobytes()).decode('ascii')
            data.append({'index':i,'embedding':value,'native_ms':round(ms,3)})
        return {'request_id':str(uuid.uuid4()),'model':'dinov3-convnext-tiny','backend':cfg.backend,
                'dimensions':768,'normalized':True,'encoding_format':encoding,'data':data,
                'timing':{'queue_ms':round(result['queue_ms'],3),'processing_ms':round(result['processing_ms'],3)}}

    @app.post('/v1/embeddings',dependencies=[Depends(authenticate)],summary='Image embedding / 单图特征')
    async def embedding(request: Request,file: UploadFile=File(...),encoding_format: Literal['float','base64']='float'):
        return encode(await infer(request,[file]),encoding_format)

    @app.post('/v1/embeddings/batch',dependencies=[Depends(authenticate)],summary='Logical image batch / 多图请求')
    async def batch(request: Request,files: list[UploadFile]=File(...),encoding_format: Literal['float','base64']='float'):
        return encode(await infer(request,files),encoding_format)

    @app.post('/v1/similarity',dependencies=[Depends(authenticate)],summary='Cosine similarity / 两图相似度')
    async def similarity(request: Request,image_a: UploadFile=File(...),image_b: UploadFile=File(...)):
        result=await infer(request,[image_a,image_b])
        return {'model':'dinov3-convnext-tiny','backend':cfg.backend,
                'cosine_similarity':float(np.clip(np.dot(*result['vectors']),-1,1)),
                'timing':{'queue_ms':round(result['queue_ms'],3),'processing_ms':round(result['processing_ms'],3)}}
    return app


app=create_app()
