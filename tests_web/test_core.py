from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from webapp.core import decode, ProcessingError, png_bytes


def encoded(im,format='PNG'):
    b=BytesIO();im.save(b,format=format);return b.getvalue()


def test_transparency_and_grayscale():
    im=Image.new('RGBA',(2,1),(0,0,0,0))
    result=decode(encoded(im));assert np.array_equal(result,[[255,255]])


def test_reject_high_bit_depth_and_multiple_frames():
    im=Image.fromarray(np.zeros((2,2),dtype=np.uint16))
    with pytest.raises(ProcessingError):decode(encoded(im))
    a=Image.new('RGB',(2,2),'white');b=Image.new('RGB',(2,2),'black')
    stream=BytesIO();a.save(stream,format='TIFF',save_all=True,append_images=[b])
    with pytest.raises(ProcessingError):decode(stream.getvalue())


def test_float32_sign_and_visual_rule():
    raw=np.array([[-3,1275,2.5]],dtype=np.float32)
    im=Image.open(BytesIO(png_bytes(raw,'laplaciano')))
    assert np.array_equal(np.asarray(im),[[3,255,3]])
    assert raw[0,0]==-3 and raw[0,1]==1275
