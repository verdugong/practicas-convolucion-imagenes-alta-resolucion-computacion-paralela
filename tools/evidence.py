"""Capturar entorno y evidencia Docker desde el anfitrion, sin mutaciones."""
import argparse
from pathlib import Path
import requests
from .common import command, environment, save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--url',default='http://localhost:8000')
    p.add_argument('--image',default='practica3:1.0')
    p.add_argument('--container',default='practica3')
    p.add_argument('--location',required=True,help='Proveedor/region o laboratorio y equipo')
    p.add_argument('--network-note',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    record=dict(host=environment(),location=a.location,network_note=a.network_note,url=a.url,
        docker_version=command(['docker','version']),docker_info=command(['docker','info']),
        runtime_config=command(['nvidia-ctk','--version']),
        image=command(['docker','image','inspect',a.image]),
        container=command(['docker','inspect',a.container]),
        running=command(['docker','ps','--no-trunc']),
        logs=command(['docker','logs',a.container]),
        container_env=command(['docker','exec',a.container,'python','-m','tools.diagnose']))
    try:
        r=requests.get(a.url+'/api/health',timeout=130)
        record['health']=dict(http_status=r.status_code,body=r.json())
    except Exception as exc:record['health_error']=str(exc)
    save_json(a.out/'deployment.json',record)
    print('Evidencia guardada en',a.out)

if __name__=='__main__':main()
