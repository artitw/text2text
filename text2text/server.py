import text2text as t2t
import numpy as np
import socket
import threading
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
  return {"result": "Hello, this Text2Text at your service."}

@app.route('/index/<action>', methods=['POST'])
def index(action):
  try:
    indexer = t2t.Server.indexer
    assert hasattr(indexer, action)
    data = request.get_json()
    input_lines = data.get("input_lines", [])
    src_lang = data.get("src_lang", "en")
    res = getattr(indexer, action)(**data)
    if action in ["search", "size"]:
      return {"result": np.array(res).tolist()}
  except Exception as e:
    return {"result": str(e)}
  return {"result": f"index/{action} performed"}

@app.route('/<transformation>', methods=['POST'])
def transform(transformation):
  try:
    assert transformation in t2t.Handler.EXPOSED_TRANSFORMERS
    data = request.get_json()
    input_lines = data.get("input_lines", [])
    src_lang = data.get("src_lang", "en")
    h = t2t.Handler(input_lines, src_lang)
    del data["input_lines"]
    del data["src_lang"]
    res = getattr(h, transformation)(**data)
    res = getattr(res, "tolist", lambda: res)()
    return {"result": res}
  except Exception as e:
    return {"result": str(e)}

class Server(object):
  def __init__(self, **kwargs):
    self.__class__.indexer = t2t.Handler().index()
    address = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
    print(f"Serving at http://{address}/")
    threading.Thread(target=app.run, kwargs=kwargs).start()
