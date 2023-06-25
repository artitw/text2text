import text2text as t2t
import numpy as np
import socket
import threading
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
  return {"result": "Hello, this Text2Text at your service."}

@app.route('/indexer/<action>', methods=['POST'])
def indexer(action):
  try:
    data = request.get_json() or {}
    input_lines = data.get("input_lines", [])
    src_lang = data.get("src_lang", "en")
    if hasattr(t2t.Server, "index"):
      index = t2t.Server.index
    else:
      t2t.Server.index = t2t.Indexer().add(input_lines)
    assert hasattr(index, action)
    res = getattr(index, action)(**data)
    if action in ["search", "size"]:
      return {"result": np.array(res).tolist()}
  except Exception as e:
    return {"result": str(e)}
  return {"result": f"indexer/{action} performed"}

@app.route('/<transformer>', methods=['POST'])
def transform(transformer):
  try:
    assert hasattr(t2t, transformer)
    data = request.get_json() or {}
    input_lines = data.get("input_lines", [])
    src_lang = data.get("src_lang", "en")
    res = getattr(t2t, transformer)(input_lines, src_lang)
    del data["input_lines"]
    del data["src_lang"]
    return {"result": res}
  except Exception as e:
    return {"result": str(e)}

class Server(object):
  def __init__(self, **kwargs):
    address = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
    print(f"Serving at http://{address}/")
    threading.Thread(target=app.run, kwargs=kwargs).start()
