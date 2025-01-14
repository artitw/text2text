import text2text as t2t
import torch
from tqdm.auto import tqdm
from openai import OpenAI
from openai.types import CreateEmbeddingResponse


class Vectorizer(object):
  def __init__(self, **kwargs):
    self.config = kwargs.get("config", {
      "model": "intfloat/e5-mistral-7b-instruct",
      "api-key": "TEXT2TEXT",
      "port": 11211,
      "task": "embed",
      "max-num-seqs": 8,
      "enforce-eager": True,
    })

  def embed(self, input_lines):
    embedder = t2t.Assistant(config=self.config)
    responses: CreateEmbeddingResponse = embedder.client.embeddings.create(
        input=input_lines,
        model=embedder.config['model'],
        encoding_format="float",
    )
    return [data.embedding for data in responses.data]

  def transform(self, input_lines, **kwargs):
    sentencize = kwargs.get("sentencize", True)
    if sentencize and input_lines != ["DUMMY"]:
      asst = t2t.Assistant()
      sentences = []
      for text in tqdm(input_lines, desc='Summarize'):
        if len(text) > 100:
          prompt = f'Summarize the following text to a single sentence:\n\n{text}'
          result = asst.chat_completion([{"role": "user",  "content": prompt}])
          sentences.append(result["message"]["content"])
        else:
          sentences.append(text)
      return self.embed(sentences)
    return self.embed(input_lines)