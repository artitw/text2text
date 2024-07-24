import os
import ollama
import psutil

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
    self.client = ollama.Client(host=self.model_url)

  def load_model(self):
    return_code = os.system(f"ollama serve & ollama pull {self.model_name}")
    if return_code != 0:
      print(f"{self.model_name} is not loading up. Maybe needs more memory. Restarting and trying again might help.")

  def chat_completion(self, messages=[{"role": "user", "content": "hello"}], stream=False, **kwargs):
    if is_port_in_use(self.port):
      return self.client.chat(model=self.model_name, messages=messages, stream=stream)
    self.load_model()
    return self.chat_completion(messages=messages, stream=stream, **kwargs)

  def transform(self, input_lines, src_lang='en', **kwargs):
    return self.chat_completion([{"role": "user", "content": input_lines}])["message"]["content"]

  completion = transform