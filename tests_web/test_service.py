"""Pruebas HTTP y cola sin NVIDIA. Simulacion explicita SOLO en tests."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import threading
import time
import numpy as np
from PIL import Image
import pytest
from webapp.app import create_app
from webapp.core import ProcessingError, decode, Limits, png_bytes
from src.python.filtros_python import crear_filtro_cpu


class TestBackend:
    __test__=False
    info=dict(backend='simulado_solo_tests',gpu='NO NVIDIA - NO EVIDENCIA DE RENDIMIENTO')
    hold=None
    started=None
    def run(self,entry,fil,k,threads):
        if self.started:self.started.set()
        if self.hold:assert self.hold.wait(5)
        return crear_filtro_cpu(fil,k).ejecutar(entry),dict(h2d_ms=0,kernel_ms=0,d2h_ms=0,
            total_ms=0,allocation_ms=0,blocks=(entry.size+threads-1)//threads)
    def close(self):pass


def image(value=1,w=17,h=19):
    b=BytesIO();Image.fromarray(np.full((h,w),value,dtype=np.uint8)).save(b,format='PNG');return b.getvalue()


@pytest.fixture
def app(tmp_path):
    a=create_app(dict(TESTING=True,RESULT_ROOT=str(tmp_path),MAX_IMAGE_BYTES=4096),TestBackend)
    yield a
    a.extensions['cleaner_stop'].set();a.extensions['gpu_worker'].close()


def post(client,data=None,**params):
    return client.post('/api/process',data=dict(image=(BytesIO(data or image()),'unsafe/../../same.png'),**params))


def test_ui_and_single_filter(app):
    with app.test_client() as c:
        assert c.get('/').status_code==200
        r=post(c,filter='nitidez',mask='3',raw='1');assert r.status_code==200
        body=r.json;assert body['metrics']['filter']=='nitidez'
        output=c.get(body['result_url']);assert output.mimetype=='image/png'
        raw=np.load(BytesIO(c.get(body['raw_url']).data),allow_pickle=False)
        assert raw.shape==(19,17)
        assert body['metrics']['backend']=='simulado_solo_tests'


@pytest.mark.parametrize('data,params,status',[
    (b'invalido',{},422),(b'x'*4097,{},413),(None,dict(filter='otro'),400),
    (None,dict(filter='nitidez',mask='9'),400),(None,dict(threads='42'),400),
    (None,dict(mask='abc'),400)])
def test_invalid_and_recovery(app,data,params,status):
    with app.test_client() as c:
        assert post(c,data,**params).status_code==status
        assert post(c).status_code==200


def test_dimensions_and_visualization():
    with pytest.raises(ProcessingError):decode(image(w=20,h=20),Limits(max_pixels=30))
    raw=np.array([[-1020,255,1275]],dtype=np.float32)
    with Image.open(BytesIO(png_bytes(raw,'laplaciano'))) as im:
        assert np.array_equal(np.array(im),[[255,255,255]])


def test_isolation(app):
    with app.test_client() as c:
        a=post(c,image(20)).json;b=post(c,image(210)).json
        assert a['request_id']!=b['request_id']
        assert c.get(a['result_url']).data!=c.get(b['result_url']).data
        assert c.get('/api/results/not-valid/result').status_code==404
        assert c.get('/api/results/'+a['request_id']+'/secret').status_code==404


def test_fifo_and_rejection(app,tmp_path):
    worker=app.extensions['gpu_worker'];worker.ready.wait(3)
    TestBackend.hold=threading.Event();TestBackend.started=threading.Event()
    try:
        dirs=[]
        for i in range(3):
            d=tmp_path/str(i);d.mkdir();dirs.append(d)
        f1=worker.submit(image(20),'media',3,256,False,dirs[0])
        assert TestBackend.started.wait(3)
        f2=worker.submit(image(210),'media',3,256,False,dirs[1])
        with pytest.raises(ProcessingError) as e:worker.submit(image(),'media',3,256,False,dirs[2])
        assert e.value.status==429
        TestBackend.hold.set();a=f1.result(3);b=f2.result(3)
        assert a['sha256_input']!=b['sha256_input']
    finally:
        TestBackend.hold.set();TestBackend.hold=TestBackend.started=None


def test_gpu_missing_has_no_cpu_fallback(tmp_path):
    def unavailable():raise RuntimeError('GPU no disponible: prueba explicita')
    app=create_app(dict(TESTING=True,RESULT_ROOT=str(tmp_path)),unavailable)
    try:
        app.extensions['gpu_worker'].ready.wait(3)
        with app.test_client() as c:
            assert c.get('/api/health').status_code==503
            assert post(c).status_code==503
    finally:
        app.extensions['cleaner_stop'].set();app.extensions['gpu_worker'].close()


def test_ttl(app):
    app.config['RESULT_TTL']=0
    with app.test_client() as c:
        body=post(c).json
        assert c.get(body['result_url']).status_code==404
