import os
import ollama
import time
import subprocess
import warnings
import platform
from tqdm.auto import tqdm

def is_sudo_available():
    try:
        # Try to run 'sudo -v' which checks if sudo is available
        subprocess.run(['sudo', '-v'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        warnings.warn(str(e))
        return False
    except FileNotFoundError as e:
        warnings.warn(str(e))
        return False

def can_use_apt():
    # Check if the OS is Linux and if it is a Debian-based distribution
    if platform.system() == "Linux":
        try:
            # Check if the apt command is available
            result = os.system("apt --version")
            return result == 0  # If the command runs successfully, return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
    return False

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

def apt_install_packages(packages, sudo=True):
    try:
        # Update the package list
        cmds = ['apt', 'update']
        if sudo:
          cmds = ['sudo']+cmds
        subprocess.run(cmds, check=True)
        
        # Install the packages
        cmds = ['apt', 'install', '-q', '-y'] + packages
        if sudo:
          cmds = ['sudo']+cmds
        subprocess.run(cmds, check=True)
        
    except subprocess.CalledProcessError as e:
        raise Exception(str(e))

class Assistant(object):
  def __init__(self, **kwargs):
    self.model_name = kwargs.get("model_name", "llama3.2")
    self.ollama_serve_proc = None
    self.sudo = kwargs.get("sudo", is_sudo_available())
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

      if can_use_apt():
        apt_install_packages(['lshw', 'curl'], self.sudo)
        pbar.update(1)
      elif platform.system() == "Windows":
        raise Exception("Windows not supported.")
      else:
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

    pbar.close()

  def model_loading(self):
    try:
      ps_result = ollama.ps()
      ls_result = ollama.list()
      if ps_result and ls_result and \
      ps_result.models and ls_result.models and \
      ps_result.models[0].model.startswith(self.model_name) and \
      ls_result.models[0].model.startswith(self.model_name):
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
    keep_alive = kwargs.get("keep_alive", -1)
    
    if schema:
      try:
        response = ollama.chat(
          model=self.model_name, 
          messages=messages, 
          format=schema.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
          options={'temperature': 0},  # Make responses more deterministic
        )

        # Use Pydantic to validate the response
        schema_response = schema.model_validate_json(response.message.content)
        return schema_response
      except Exception as e:
        warnings.warn(str(e))
        warnings.warn(f"Schema extraction failed for {messages}")
        default_schema = schema()
        warnings.warn(f"Returning schema with default values: {vars(default_schema)}")
        return default_schema
    return ollama.chat(
      model=self.model_name, 
      messages=messages, 
      stream=stream, 
      keep_alive=keep_alive
    )

  def embed(self, texts, **kwargs):
    while self.model_loading(): time.sleep(1)
    keep_alive = kwargs.get("keep_alive", -1)
    return ollama.embed(
      model=self.model_name, 
      input=texts, 
      keep_alive=keep_alive
    ).get("embeddings", [])

  def transform(self, input_lines, src_lang='en', **kwargs):
    return self.chat_completion([{"role": "user", "content": input_lines}])["message"]["content"]

  completion = transform