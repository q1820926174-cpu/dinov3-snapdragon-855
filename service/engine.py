"""One warm native worker, bounded admission and cancellation-safe pipe framing."""
import asyncio
import io
import os
import struct
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class BusyError(Exception):pass
class UnavailableError(Exception):pass
class InvalidImageError(Exception):pass


@dataclass(frozen=True)
class Settings:
    backend: str='cpu'
    model_path: str='/models/model'
    executable: str='/usr/local/bin/dinov3_embed_server'
    threads: int=4
    queue_size: int=8
    timeout: float=30
    startup_timeout: float=60
    max_body_bytes: int=8*1024*1024
    max_batch: int=8
    max_pixels: int=20_000_000
    api_key: str=''

    @classmethod
    def from_env(cls):
        result=cls(backend=os.getenv('DINO_BACKEND','cpu'),model_path=os.getenv('DINO_MODEL_PATH','/models/model'),
            executable=os.getenv('DINO_EXECUTABLE','/usr/local/bin/dinov3_embed_server'),
            threads=int(os.getenv('DINO_NUM_THREADS','4')),queue_size=int(os.getenv('DINO_MAX_QUEUE','8')),
            timeout=float(os.getenv('DINO_TIMEOUT_SECONDS','30')),
            startup_timeout=float(os.getenv('DINO_STARTUP_TIMEOUT_SECONDS','60')),
            max_body_bytes=int(os.getenv('DINO_MAX_BODY_BYTES',str(8*1024*1024))),
            max_batch=int(os.getenv('DINO_MAX_BATCH','8')),
            max_pixels=int(os.getenv('DINO_MAX_IMAGE_PIXELS','20000000')),api_key=os.getenv('DINO_API_KEY',''))
        if result.backend not in ('cpu','gpu16s'):raise ValueError('DINO_BACKEND must be cpu or gpu16s')
        if not 1<=result.threads<=64 or not 1<=result.queue_size<=128:raise ValueError('invalid threads/queue size')
        if min(result.timeout,result.startup_timeout,result.max_body_bytes,result.max_batch,result.max_pixels)<=0:
            raise ValueError('limits must be positive')
        return result


def prepare_image(data: bytes,max_pixels: int)->bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width*image.height>max_pixels:raise InvalidImageError('image pixel limit exceeded')
                if image.format not in ('JPEG','PNG','WEBP','BMP'):raise InvalidImageError('unsupported image format')
                rgb=ImageOps.exif_transpose(image).convert('RGB')
                return ImageOps.pad(rgb,(224,224),method=Image.Resampling.BICUBIC,
                                    color=(114,114,114),centering=(.5,.5)).tobytes()
    except InvalidImageError:raise
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombWarning,Image.DecompressionBombError) as error:
        raise InvalidImageError('invalid or oversized image') from error


class Engine:
    def __init__(self,settings: Settings):
        self.settings=settings
        self.queue=asyncio.Queue(maxsize=settings.queue_size)
        self.process=None
        self.task=None
        self.failure=None
        self.completed_images=0
        self.active=False

    @property
    def ready(self):
        return self.process is not None and self.process.returncode is None and self.failure is None

    async def start(self):
        for suffix in ('.param','.bin'):
            if not Path(self.settings.model_path+suffix).is_file():raise RuntimeError('missing model file '+suffix)
        env=os.environ.copy();env.update(SCV41_NCNN_STRICT_GPU='1',DINO_NUM_THREADS=str(self.settings.threads))
        self.process=await asyncio.create_subprocess_exec(self.settings.executable,self.settings.model_path,
            self.settings.backend,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,env=env)
        try:
            header=await asyncio.wait_for(self.process.stdout.readexactly(4),self.settings.startup_timeout)
            if header!=b'RDY1':raise RuntimeError('invalid native protocol header')
            warm=bytes([114])*(224*224*3)
            for _ in range(3):await self._native(warm)
        except BaseException:
            await self._stop_process()
            raise
        self.task=asyncio.create_task(self._work())

    async def _native(self,rgb):
        self.process.stdin.write(rgb)
        await asyncio.wait_for(self.process.stdin.drain(),self.settings.timeout)
        data=await asyncio.wait_for(self.process.stdout.readexactly(768*4+8),self.settings.timeout)
        vector=np.frombuffer(data[:768*4],dtype='<f4').copy()
        norm=float(np.linalg.norm(vector))
        native_ms=struct.unpack('<d',data[768*4:])[0]
        if not np.isfinite(vector).all() or norm<=1e-12 or not np.isfinite(native_ms):
            raise RuntimeError('invalid native embedding')
        vector/=norm
        return vector,native_ms

    async def infer(self,images):
        if not self.ready:raise UnavailableError('native worker unavailable')
        future=asyncio.get_running_loop().create_future()
        try:self.queue.put_nowait((images,future,time.perf_counter()))
        except asyncio.QueueFull as error:raise BusyError('inference queue is full') from error
        try:return await asyncio.wait_for(asyncio.shield(future),self.settings.timeout)
        except BaseException:
            # Never cancel an in-flight read: doing so corrupts the next pipe response.
            future.cancel()
            raise

    async def _work(self):
        while True:
            images,future,queued=await self.queue.get()
            try:
                if future.cancelled():continue
                self.active=True
                start=time.perf_counter();vectors=[];native=[]
                for data in images:
                    rgb=await asyncio.to_thread(prepare_image,data,self.settings.max_pixels)
                    vector,ms=await self._native(rgb)
                    vectors.append(vector);native.append(ms)
                self.completed_images+=len(vectors)
                result={'vectors':vectors,'native_ms':native,'queue_ms':(start-queued)*1000,
                        'processing_ms':(time.perf_counter()-start)*1000}
                if not future.done():future.set_result(result)
            except InvalidImageError as error:
                if not future.done():future.set_exception(error)
            except asyncio.CancelledError:
                if not future.done():future.set_exception(UnavailableError('service shutting down'))
                raise
            except Exception:
                self.failure='native worker failed'
                if not future.done():future.set_exception(UnavailableError(self.failure))
                await self._stop_process()
                self._drain()
                return
            finally:
                self.active=False;self.queue.task_done()

    def _drain(self):
        while not self.queue.empty():
            _,future,_=self.queue.get_nowait()
            if not future.done():future.set_exception(UnavailableError('native worker unavailable'))
            self.queue.task_done()

    async def _stop_process(self):
        if self.process is not None:
            if self.process.returncode is None:
                self.process.terminate()
                try:await asyncio.wait_for(self.process.wait(),3)
                except asyncio.TimeoutError:self.process.kill();await self.process.wait()

    async def close(self):
        self.failure='service stopped'
        if self.task is not None:
            self.task.cancel()
            try:await self.task
            except asyncio.CancelledError:pass
        self._drain()
        await self._stop_process()
