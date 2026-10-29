import asyncio,base64,io,os,sys,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from service.app import create_app
from service.engine import BusyError,Engine,Settings,UnavailableError


def image(value=20):
    output=io.BytesIO();Image.new('RGB',(224,224),(value,30,40)).save(output,format='PNG');return output.getvalue()


class Fixture:
    def __init__(self):
        self.temp=tempfile.TemporaryDirectory()
        model=Path(self.temp.name)/'fake_native.py'
        model.write_text((Path(__file__).parent/'fake_native.py').read_text())
        for suffix in ['.param','.bin']:Path(str(model)+suffix).write_bytes(b'')
        self.settings=Settings(executable=sys.executable,model_path=str(model))
    def close(self):self.temp.cleanup()


class APITests(unittest.TestCase):
    def setUp(self):self.fixture=Fixture()
    def tearDown(self):self.fixture.close()
    def test_embedding_batch_similarity_and_auth(self):
        cfg=replace(self.fixture.settings,api_key='test-only-key')
        with TestClient(create_app(cfg)) as client:
            self.assertEqual(client.get('/health/ready').status_code,200)
            self.assertEqual(client.post('/v1/embeddings',files={'file':('x.png',image())}).status_code,401)
            self.assertEqual(client.post('/v1/embeddings',files={'file':('x.png',image())},
                headers={'Authorization':b'Bearer \xe9'}).status_code,401)
            client.headers['Authorization']='Bearer test-only-key'
            response=client.post('/v1/embeddings',files={'file':('x.png',image())})
            self.assertEqual(response.status_code,200,response.text)
            body=response.json();v=np.array(body['data'][0]['embedding'])
            self.assertEqual(v.shape,(768,));self.assertAlmostEqual(np.linalg.norm(v),1,places=6)
            response=client.post('/v1/embeddings/batch?encoding_format=base64',
                files=[('files',('a.png',image(20))),('files',('b.png',image(200)))])
            self.assertEqual(response.status_code,200,response.text)
            data=response.json()['data'];self.assertEqual(len(data),2)
            decoded=np.frombuffer(base64.b64decode(data[0]['embedding']),dtype='<f4')
            np.testing.assert_allclose(decoded,v,atol=1e-7)
            response=client.post('/v1/similarity',files=[('image_a',('a.png',image())),('image_b',('b.png',image()))])
            self.assertEqual(response.status_code,200,response.text)
            self.assertAlmostEqual(response.json()['cosine_similarity'],1,places=6)
    def test_bad_image_does_not_poison_worker(self):
        with TestClient(create_app(self.fixture.settings)) as client:
            self.assertEqual(client.post('/v1/embeddings',files={'file':('bad.png',b'bad')}).status_code,422)
            self.assertEqual(client.post('/v1/embeddings',files={'file':('ok.png',image())}).status_code,200)
    def test_body_limit_declared_and_streamed(self):
        with TestClient(create_app(replace(self.fixture.settings,max_body_bytes=128))) as client:
            response=client.post('/v1/embeddings',content=b'x'*129)
            self.assertEqual(response.status_code,413)
            response=client.post('/v1/embeddings',content=iter([b'x'*80,b'y'*80]))
            self.assertEqual(response.status_code,413)
    def test_native_crash_marks_unhealthy(self):
        with patch.dict(os.environ,{'DINO_TEST_CRASH':'1'}):
            with TestClient(create_app(self.fixture.settings)) as client:
                self.assertEqual(client.post('/v1/embeddings',files={'file':('x.png',image())}).status_code,503)
                self.assertEqual(client.get('/health/ready').status_code,503)
    def test_model_missing_fails_startup(self):
        with self.assertRaises(RuntimeError):
            with TestClient(create_app(replace(self.fixture.settings,model_path='/missing/model'))):pass


class QueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):self.fixture=Fixture();self.engine=None
    async def asyncTearDown(self):
        if self.engine:await self.engine.close()
        self.fixture.close()
    async def test_bounded_queue(self):
        with patch.dict(os.environ,{'DINO_TEST_DELAY':'.1'}):
            self.engine=Engine(replace(self.fixture.settings,queue_size=1));await self.engine.start()
            first=asyncio.create_task(self.engine.infer([image()]));await asyncio.sleep(.025)
            second=asyncio.create_task(self.engine.infer([image()]));await asyncio.sleep(.01)
            with self.assertRaises(BusyError):await self.engine.infer([image()])
            await asyncio.gather(first,second)
    async def test_timed_out_batch_does_not_misalign_following_response(self):
        with patch.dict(os.environ,{'DINO_TEST_DELAY':'.1'}):
            self.engine=Engine(replace(self.fixture.settings,timeout=.2));await self.engine.start()
            with self.assertRaises(asyncio.TimeoutError):await self.engine.infer([image(1),image(2),image(3)])
            await self.engine.queue.join()
            self.assertTrue(self.engine.ready)
            result=await self.engine.infer([image(200)])
            vector=result['vectors'][0]
            self.assertGreater(vector[0],vector[1])
    async def test_close_reaps_child(self):
        self.engine=Engine(self.fixture.settings);await self.engine.start()
        await self.engine.close();self.assertIsNotNone(self.engine.process.returncode)


if __name__=='__main__':unittest.main()
