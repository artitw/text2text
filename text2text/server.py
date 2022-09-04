import text2text as t2t
import socket
import threading
from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello():
  return "Hello, this is the Text2Text server up and running."

@app.route('/<transformation>', methods=['POST'])
def transform(transformation):
  assert transformation in t2t.Handler.EXPOSED_TRANSFORMERS
  data = request.get_json()
  input_lines = data.get("input_lines", [])
  src_lang = data.get("src_lang", "en")
  h = t2t.Handler(input_lines, src_lang)
  del data["input_lines"]
  del data["src_lang"]
  return {"result": getattr(h, transformation)(**data)}

class Server(object):
  def __init__(self):
    address = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
    print(f"Serving at http://{address}/")
    threading.Thread(target=app.run, kwargs={'host':'0.0.0.0','port':80}).start()
