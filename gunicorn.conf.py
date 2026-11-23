# Un proceso HTTP y un propietario CUDA. No incrementar workers sin redisenar la cola.
bind = '0.0.0.0:8000'
workers = 1
worker_class = 'gthread'
threads = 8
timeout = 300
graceful_timeout = 300
preload_app = False
reload = False
accesslog = '-'
errorlog = '-'
capture_output = True
