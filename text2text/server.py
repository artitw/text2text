import text2text as t2t
import numpy as np
import socket
import threading
from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello():
  return {"result": "Hello, this Text2Text at your service."}

@app.route('/Indexer/<action>', methods=['POST'])
def indexer(action):
  try:
    if not hasattr(t2t.Server, "index"):
      t2t.Server.index = t2t.Indexer().transform([])
    index = t2t.Server.index
    assert hasattr(index, action)
    data = request.get_json() or {}
    res = getattr(index, action)(**data)
    if action in ["search", "size"]:
      return {"result": np.array(res).tolist()}
    elif action in ["retrieve"]:
      return {"result": res}
  except Exception as e:
    return {"result": str(e)}
  return {"result": f"Indexer/{action} performed"}

@app.route('/<transformer>', methods=['POST'])
def transform(transformer):
  try:
    assert hasattr(t2t, transformer)
    transformer_instance = transformer.lower()
    if not hasattr(t2t.Server, transformer_instance):
      setattr(t2t.Server, transformer_instance, getattr(t2t, transformer)())
    data = request.get_json() or {}
    res = getattr(t2t.Server, transformer_instance).transform(**data)
    return {"result": res}
  except Exception as e:
    return {"result": str(e)}

class Server(object):
  def __init__(self, **kwargs):
    address = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
    print(f"Serving at http://{address}/")
    threading.Thread(target=app.run, kwargs=kwargs).start()
