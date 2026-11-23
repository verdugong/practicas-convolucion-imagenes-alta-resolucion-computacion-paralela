"""Prueba REAL de GPU: crea contexto, compila kernel y ejecuta matriz pequena."""
import json
import numpy as np
from webapp.core import GPUBackend
from .common import environment


def main():
    backend=GPUBackend()
    try:
        entry=np.arange(1,10,dtype=np.float32).reshape(3,3)
        result,times=backend.run(entry,'media',3,256)
        expected=np.array([[12,21,16],[27,45,33],[24,39,28]],dtype=np.float32)/np.float32(9)
        err=float(np.abs(result-expected).max())
        if err>0.001:raise AssertionError('GPU no coincide con matriz manual.')
        print(json.dumps(dict(environment=environment(),gpu=backend.info,
            kernel_executed=True,error_max=err,times=times),indent=2))
    finally:backend.close()

if __name__=='__main__':main()
