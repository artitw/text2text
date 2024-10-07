import text2text as t2t

import urllib.parse
import urllib.request

import warnings

def is_valid_url(url):
  try:
    result = urllib.parse.urlparse(url)
    return all([result.scheme, result.netloc])
  except Exception:
    return False

class RagAssistant(t2t.Assistant):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    schema = kwargs.get("schema", None)
    texts = kwargs.get("texts", [])
    urls = kwargs.get("urls", [])
    input_lines = []
    for u in urls:
      if is_valid_url(u):
        try:
          with urllib.request.urlopen(u) as f:
            texts.append(f.read())
        except Exception as e:
          warnings.warn(f"Skipping URL with errors: {u}")
      else:
        warnings.warn(f"Skipping invalid URL: {u}")

    if schema:
      for t in texts:
        res = t2t.Assistant.chat_completion(self, [{"role": "user",  "content": t}], schema=schema)
        res = "\n".join(f'{k}: {v}' for k,v in vars(res).items())
        input_lines.append(res)
    else:
      input_lines = texts

    self.index = t2t.Indexer().transform(input_lines, encoders=[t2t.Vectorizer()])

  def chat_completion(self, messages=[{"role": "user", "content": "hello"}], stream=False, schema=None, **kwargs):
    k = kwargs.get("k", 3)
    query = messages[-1]["content"]
    docs = self.index.retrieve([query], k=k)[0]
    grounding_information = "\n\n".join(docs) + "\n\n"
    messages[-1] = {"role": "user", "content": grounding_information+query}
    return t2t.Assistant.chat_completion(self, messages=messages, stream=stream, schema=schema, **kwargs)
