import os
import ollama
import psutil
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

def is_port_in_use(port):
  for conn in psutil.net_connections():
    if conn.status == psutil.CONN_LISTEN and conn.laddr.port == int(port):
      return True
  return False


class Assistant(object):
  def __init__(self, **kwargs):
    self.host = kwargs.get("host", "http://localhost")
    self.port = kwargs.get("port", 11434)
    self.model_url = f"{self.host}:{self.port}"
    self.model_name = kwargs.get("model_name", "llama3.1")
    return_code = os.system("curl -fsSL https://ollama.com/install.sh | sh")
    if return_code != 0:
      print("Cannot install ollama.")
    self.load_model()
    self.client = Ollama(model=self.model_name, request_timeout=120.0)

  def load_model(self):
    return_code = os.system("sudo service ollama stop")
    return_code = os.system(f"ollama serve & ollama pull {self.model_name}")
    if return_code != 0:
      print(f"{self.model_name} is not loading up. Restarting and trying again might help. Maybe needs more memory.")

  def chat_completion(self, messages=[{"role": "user", "content": "hello"}], stream=False, schema=None, **kwargs):
    if is_port_in_use(self.port):
      msgs = [ChatMessage(**m) for m in messages]
      if stream:
        return self.client.stream_chat(messages=msgs)
      if schema:
        return self.client.as_structured_llm(schema).chat(messages=msgs).raw
      return self.client.chat(messages=msgs).raw
    self.load_model()
    return self.chat_completion(messages=messages, stream=stream, **kwargs)

  def transform(self, input_lines, src_lang='en', **kwargs):
    return self.chat_completion([{"role": "user", "content": input_lines}])["message"]["content"]

  completion = transform