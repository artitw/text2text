import os
import ollama
import time
import subprocess
import warnings

from tqdm.auto import tqdm
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

def ollama_version():
  try:
    result = subprocess.check_output(["ollama", "-v"], stderr=subprocess.STDOUT).decode("utf-8")
    if result.startswith("ollama version "):
      return result.replace("ollama version ", "")
  except Exception as e:
    pass
  return ""

def run_sh(script_string):
  try:
    process = subprocess.Popen(
        ['sh'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Treat input/output as text
    )
    output, error = process.communicate(input=script_string)

    if process.returncode == 0:
        return output
    else:
        return error

  except Exception as e:
      return str(e)

class Assistant(object):
  def __init__(self, **kwargs):
    self.host = kwargs.get("host", "http://localhost")
    self.port = kwargs.get("port", 11434)
    self.model_url = f"{self.host}:{self.port}"
    self.model_name = kwargs.get("model_name", "llama3.2")
    self.schema_timeout = kwargs.get("schema_timeout", 120.0)
    self.ollama_serve_proc = None
    self.load_model()

  def __del__(self):
    try:
      if ollama_version():
        ollama.delete(self.model_name)
      if self.ollama_serve_proc:
        self.ollama_serve_proc.kill()
        self.ollama_serve_proc = None
    except Exception as e:
      warnings.warn(str(e))

  def load_model(self):
    pbar = tqdm(total=6, desc='Model Setup')
    if not ollama_version():
      self.__del__()
      pbar.update(1)

      return_code = os.system("sudo apt install -q -y lshw")
      if return_code != 0:
        raise Exception("Cannot install lshw.")
      pbar.update(1)

      result = os.system(
        "curl -fsSL https://ollama.com/install.sh | sh"
      )
      if result != 0:
        raise Exception("Cannot install ollama")
      pbar.update(1)

      self.ollama_serve_proc = subprocess.Popen(["ollama", "serve"])
      time.sleep(1)
      pbar.update(1)

      if not ollama_version():
        raise Exception("Cannot serve ollama")
      pbar.update(1)
    else:
      pbar.update(5)
    
    result = ollama.pull(self.model_name)
    if result["status"] == "success":
      ollama_run_proc = subprocess.Popen(["ollama", "run", self.model_name])
      pbar.update(1)
    else:
      raise Exception(f"Did not pull {self.model_name}. Try restarting.")
    
    self.client = ollama.Client(host=self.model_url)
    self.structured_client = Ollama(
      model=self.model_name, 
      request_timeout=self.schema_timeout
    )

    pbar.close()

  def model_loading(self):
    try:
      ps_result = ollama.ps()
      ls_result = ollama.list()
      if ps_result and ls_result and \
      ps_result.get("models", []) and ls_result.get("models", []) and \
      ps_result.get("models")[0].get("name", "").startswith(self.model_name) and \
      ls_result.get("models")[0].get("name", "").startswith(self.model_name):
        return False
    except Exception as e:
      warnings.warn(str(e))
    warnings.warn("Model not loaded. Retrying...")
    self.load_model()
    return True
        
  def chat_completion(self, messages = [{"role": "user", "content": "hello"}], **kwargs):
    while self.model_loading(): time.sleep(1)
    stream = kwargs.get("stream", False)
    schema = kwargs.get("schema", None)
    
    if schema:
      try:
        msgs = [ChatMessage(**m) for m in messages]
        return self.structured_client.as_structured_llm(schema).chat(messages=msgs).raw
      except Exception as e:
        warnings.warn(str(e))
        warnings.warn(f"Schema extraction failed for {messages}")
        default_schema = schema()
        warnings.warn(f"Returning schema with default values: {vars(default_schema)}")
        return default_schema
    return self.client.chat(model=self.model_name, messages=messages, stream=stream)

  def embed(self, texts):
    while self.model_loading(): time.sleep(1)
    return self.client.embed(model=self.model_name, input=texts).get("embeddings", [])

  def transform(self, input_lines, src_lang='en', **kwargs):
    return self.chat_completion([{"role": "user", "content": input_lines}])["message"]["content"]

  completion = transform