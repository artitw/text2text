import torch
import numpy as np
import text2text as t2t
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import sklearn.preprocessing as preprocessing

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
  last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
  return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class Vectorizer(t2t.Transformer):

  def __init__(self, pretrained_model='intfloat/multilingual-e5-large', batch_size=0):
    self.__class__.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    self.__class__.model = AutoModel.from_pretrained(pretrained_model)
    self.__class__.batch_size = batch_size

  def batch_embed(self, input_lines):
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    encoder_inputs = tokenizer(input_lines, max_length=512, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
      outputs = model(**encoder_inputs)
    embeddings = average_pool(outputs.last_hidden_state, encoder_inputs['attention_mask'])
    return preprocessing.normalize(embeddings, norm='l2')

  def transform(self, input_lines, src_lang='en', batch_process=False, **kwargs):
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    batch_size = self.__class__.batch_size
    if batch_size > 0:
      embeddings = None
      for i in range(0, len(input_lines), batch_size):
        lines = input_lines[i:i+batch_size]
        this_batch = self.batch_embed(lines)
        if embeddings is None: embeddings = this_batch
        else: embeddings = np.concatenate((embeddings, this_batch), axis=0)
      return np.array(embeddings)
    else:
      return np.array(self.batch_embed(input_lines))
