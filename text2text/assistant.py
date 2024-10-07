import os
import ollama
import psutil
import time
import subprocess
import warnings

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

def ollama_version():
  try:
    result = subprocess.check_output(["ollama", "-v"], stderr=subprocess.STDOUT).decode("utf-8")
    if result.startswith("ollama version "):
      return result.replace("ollama version ", "")
  except Exception as e:
    warnings.warn(str(e))
  return ""

class Assistant(object):
  def __init__(self, **kwargs):
    self.host = kwargs.get("host", "http://localhost")
    self.port = kwargs.get("port", 11434)
    self.model_url = f"{self.host}:{self.port}"
    self.model_name = kwargs.get("model_name", "llama3.2")
    self.ollama_serve_proc = None
    self.load_model()
    self.client = ollama.Client(host=self.model_url)
    self.structured_client = Ollama(model=self.model_name, request_timeout=120.0)

  def __del__(self):
    ollama.delete(self.model_name)
    self.ollama_serve_proc.kill()

  def load_model(self):
    if not ollama_version():
      if self.ollama_serve_proc:
        self.ollama_serve_proc.kill()
        self.ollama_serve_proc = None

      return_code = os.system("sudo apt install -q -y lshw")
      if return_code != 0:
        raise Exception("Cannot install lshw.")

      return_code = os.system("curl -fsSL https://ollama.com/install.sh | sh")
      if return_code != 0:
        raise Exception("Cannot install ollama.")

      return_code = os.system("sudo systemctl enable ollama")
      if return_code != 0:
        raise Exception("Cannot enable ollama.")

      self.ollama_serve_proc = subprocess.Popen(["ollama", "serve"])
      time.sleep(1)

      result = subprocess.check_output(["ollama", "-v"], stderr=subprocess.STDOUT).decode("utf-8")
      if not result.startswith("ollama version"):
        raise Exception(result)
      
    result = ollama.pull(self.model_name)
    if result["status"] != "success":
      raise Exception(f"Did not pull {self.model_name}.")
        
  def chat_completion(self, messages=[{"role": "user", "content": "hello"}], stream=False, schema=None, **kwargs):
    try:
      result = ollama.ps()
      if not result or not result.get("models"):
        result = ollama.pull(self.model_name)
        if result["status"] == "success":
          return self.chat_completion(messages=messages, stream=stream, **kwargs)
        raise Exception(f"Did not pull {self.model_name}. Try restarting.")
    except Exception as e:
      warnings.warn(str(e))
      warnings.warn("Retrying...")
      self.load_model()
      return self.chat_completion(messages=messages, stream=stream, **kwargs)
    
    if schema:
      msgs = [ChatMessage(**m) for m in messages]
      return self.structured_client.as_structured_llm(schema).chat(messages=msgs).raw
    return self.client.chat(model=self.model_name, messages=messages, stream=stream)

  def embed(self, texts):
    return ollama.embed(model=self.model_name, input=texts).get("embeddings", [])

  def transform(self, input_lines, src_lang='en', **kwargs):
    return self.chat_completion([{"role": "user", "content": input_lines}])["message"]["content"]

  completion = transform