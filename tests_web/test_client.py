"""Contrato de cliente: se descarga PNG, se guardan errores y no solo JSON."""
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import numpy as np
from tools.client import one


class Reply:
    def __init__(self,body=None,status=200,content=b'PNG_COMPLETO'):
        self.body,self.status_code,self.content=body,status,content
        self.ok=status<400;self.text='respuesta'
    def json(self):return self.body
    def raise_for_status(self):
        if not self.ok:raise RuntimeError(self.status_code)


def test_full_image_download(tmp_path):
    p=tmp_path/'imagen.pgm';Image.fromarray(np.zeros((3,5),dtype=np.uint8)).save(p)
    body=dict(request_id='id',result_url='/api/results/id/result',metrics=dict(kernel_ms=1))
    with patch('tools.client.requests.post',return_value=Reply(body)) as post, \
         patch('tools.client.requests.get',return_value=Reply()) as get:
        row,image,raw=one('http://test',p,'media',3,256)
    assert row['status']=='ok' and image==b'PNG_COMPLETO'
    assert get.call_args.args[0]=='http://test/api/results/id/result'
    assert row['client_total_ms']>=0 and row['width']==5 and raw is None


def test_rejected_request_is_recorded(tmp_path):
    p=tmp_path/'imagen.pgm';Image.fromarray(np.zeros((3,5),dtype=np.uint8)).save(p)
    with patch('tools.client.requests.post',return_value=Reply(dict(request_id='id',message='cola llena'),429)):
        row,image,raw=one('http://test',p,'media',3,256)
    assert row['status']=='error' and row['http_status']==429 and row['error']=='cola llena'
    assert row['client_total_ms']>=0 and image is None
