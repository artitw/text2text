import torch
import numpy as np
import text2text as t2t
from transformers import AutoTokenizer, AutoModel
import sklearn.preprocessing as preprocessing

def mean_pooling(model_output, attention_mask):
  token_embeddings = model_output[0]
  input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
  return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Vectorizer(t2t.Transformer):

  def __init__(self, pretrained_model='sentence-transformers/all-mpnet-base-v2', batch_size=0):
    self.__class__.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    self.__class__.model = AutoModel.from_pretrained(pretrained_model)
    self.__class__.batch_size = batch_size

  def batch_embed(self, input_lines):
    encoder_inputs = self.__class__.tokenizer(input_lines, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
      model_output = self.__class__.model(**encoder_inputs)
    X = mean_pooling(model_output, encoder_inputs['attention_mask'])
    return preprocessing.normalize(X, norm='l2')

  def transform(self, input_lines, src_lang='en', batch_process=False, **kwargs):
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
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
