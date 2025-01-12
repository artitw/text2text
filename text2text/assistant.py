import os
import gc
import openai
import time
import psutil
import shlex
import signal
import platform
import requests
import subprocess
import warnings
import torch
import vllm
from tqdm.auto import tqdm
from openai import OpenAI


def can_use_apt():
  # Check if the OS is Linux and if it is a Debian-based distribution
  if platform.system() == "Linux":
    try:
      # Check if the apt command is available
      result = os.system("apt --version")
      return result == 0  # If the command runs successfully, return True
    except Exception as e:
      warnings.warn(str(e))
      return False
  return False

def kill_processes(keyword):
  pids = []
  # Iterate over all running processes
  for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
      # Check if the process name or command line contains 'vllm'
      if keyword in ''.join(proc.info['cmdline']):
        # Terminate the process
        proc.terminate()  # or os.kill(proc.info['pid'], signal.SIGTERM)
        proc.wait()  # Wait for the process to terminate
        pids.append(proc.info['pid'])
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
      pass
  return pids

class Assistant(object):
  def __init__(self, **kwargs):
    self.config = kwargs.get("config", {
      "model": "unsloth/Llama-3.2-3B-Instruct",
      "api-key": "TEXT2TEXT",
      "port": 11434,
      "max-model-len": 32000,
      "dtype": "half",
      "task": "generate",
    })
    self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    self.min_device_memory_gb = kwargs.get("min_device_memory_gb", 8)
    self.server_proc = None
    self.load_model()
    self.client = OpenAI(
      base_url = f"http://localhost:{self.config['port']}/v1",
      api_key = self.config['api-key'],
    )

  def __del__(self):
    try:
      if self.server_proc:
        self.server_proc.kill()
        self.server_proc = None
    except Exception as e:
      warnings.warn(str(e))
    gc.collect()
    torch.cuda.empty_cache()

  def is_server_up(self):
    try:
      url = f'http://localhost:{self.config["port"]}/v1/models'
      headers = {"Authorization": f"Bearer {self.config['api-key']}"}
      res = requests.get(url, headers=headers)
      if res.status_code == 200:
        data = res.json().get("data", [])
        if not data:
          warnings.warn("No models found")
          return False
        model_name = data[0].get("id", "")
        if model_name == self.config["model"]:
          return True
        warnings.warn(f'Running "{model_name}" does not match {self.config["model"]}')
      else:
        warnings.warn(res.text)
    except Exception as e:
      warnings.warn(str(e))
    return False

  def set_available_device(self, num_tries=0):
    if num_tries > 3:
      warnings.warn(f"{num_tries} times setting device. Aborting.")
      return

    memory_cuda = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    memory_cpu = psutil.virtual_memory().available / (1024 ** 3)

    if self.config["device"] == "cuda" and memory_cuda < self.min_device_memory_gb:
      warnings.warn(f"{self.config['device']} {memory_cuda}GB RAM free is less than {self.min_device_memory_gb}GB specified.")
      if memory_cuda+memory_cpu >= self.min_device_memory_gb:
        self.config["cpu-offload-gb"] = memory_cpu
        warnings.warn(f"{memory_cpu}GB cpu offloading")
      else:
        self.config["device"] = "cpu"
        warnings.warn(f"Set device to {self.config['device']}")
        self.set_available_device(num_tries=num_tries+1)
    elif memory_cpu < self.min_device_memory_gb:
      warnings.warn(f"{self.config['device']} {memory_cpu}GB RAM free is less than {self.min_device_memory_gb}GB specified.")
      pids = kill_processes("vllm")
      gc.collect()
      torch.cuda.empty_cache()
      warnings.warn(f"Killed processes {pids}")
      self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
      warnings.warn(f"Set device to {self.config['device']}")
      self.set_available_device(num_tries=num_tries+1)

  def serve_model(self):
    self.set_available_device()
    args_strs = [f"--{k} {self.config[k]}" for k in self.config] 
    args_str = ' '.join(args_strs)
    cmd_str = f"python -m vllm.entrypoints.openai.api_server {args_str}"
    try:
      self.server_proc = subprocess.Popen(
        shlex.split(cmd_str), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1,
      )
    except Exception as e:
      warnings.warn(str(e))

  def wait_for_startup(self):
    while True:
      output = self.server_proc.stdout.readline()
      if self.server_proc.poll() is not None:
        raise Exception(output)
      if "Application startup complete" in output:
        break
      time.sleep(1.0)

  def load_model(self):
    pbar = tqdm(total=5, desc=f'Model Setup ({self.config["port"]})')
    if self.is_server_up():
      pbar.update(5)
    else:
      pbar.update(1)
      self.__del__()
      pbar.update(1)
      if not can_use_apt():
        warnings.warn("Text2Text not tested on this system.")
      self.serve_model()
      pbar.update(1)
      self.wait_for_startup()
      pbar.update(1)
      if not self.is_server_up():
        raise Exception("vLLM server not found after startup")
      pbar.update(1)
    pbar.close()
        
  def chat_completion(self, messages = [{"role": "user", "content": "hello"}], **kwargs):
    stream = kwargs.get("stream", False)
    schema = kwargs.get("schema", None)
    
    if schema:
      try:
        completion = self.client.beta.chat.completions.parse(
            model=self.config["model"],
            messages=messages,
            response_format=schema,
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        schema_response = completion.choices[0].message.parsed
        return schema_response
      except Exception as e:
        warnings.warn(str(e))
        warnings.warn(f"Schema extraction failed for {messages}")
        default_schema = schema()
        warnings.warn(f"Returning schema with default values: {vars(default_schema)}")
        return default_schema
    return self.client.chat.completions.create(
      model=self.config["model"], 
      messages=messages, 
      stream=stream, 
    )

  def transform(self, input_lines, src_lang='en', **kwargs):
    return self.chat_completion([{"role": "user", "content": input_lines}])

  completion = transform