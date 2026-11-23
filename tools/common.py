import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['request_id','mode','experiment','filter','width','height','mask','threads','blocks',
          'repetition','round','concurrency','phase','h2d_ms','kernel_ms','d2h_ms','total_ms',
          'allocation_ms','decode_prepare_ms','encode_ms','write_ms','processing_ms',
          'server_processing_ms','queue_ms','client_total_ms','round_ms','successful_per_second',
          'unit','status','http_status','error','sha256_input','sha256_float32','environment']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def command(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,errors='replace',timeout=30)
        return dict(command=args,returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    except (OSError,subprocess.TimeoutExpired) as exc:
        return dict(command=args,error=str(exc))

def environment():
    info=dict(time_utc=datetime.now(timezone.utc).isoformat(),hostname=socket.gethostname(),
        platform=platform.platform(),cpu=platform.processor(),cpu_count=os.cpu_count(),
        python=platform.python_version(),commit=command(['git','rev-parse','HEAD']),
        image_commit=os.getenv('CODE_COMMIT','no_registrado'),
        nvidia=command(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv']),
        nvcc=command(['nvcc','--version']),pip_freeze=command([os.sys.executable,'-m','pip','freeze']),
        ram=command(['free','-b']),lscpu=command(['lscpu','--json']))
    if platform.system()=='Windows':
        info['windows_hardware']=command(['powershell','-NoProfile','-Command',
            'Get-CimInstance Win32_ComputerSystem | Select-Object Model,TotalPhysicalMemory | ConvertTo-Json'])
        info['windows_cpu']=command(['powershell','-NoProfile','-Command',
            'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json'])
    return info

def save_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def writer(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists(): raise FileExistsError('No sobrescribir evidencia: '+str(p))
    f=p.open('w',newline='',encoding='utf-8')
    w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();f.flush()
    return f,w
