import text2text as t2t
import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing as preprocessing
import warnings

class Indexer(t2t.Transformer):

  def __init__(self, **kwargs):
    self.input_lines = [] 
    self.encoders = kwargs.get("encoders", [t2t.Tfidfer()])
    columns = ["document", "embedding"]
    self.corpus = pd.DataFrame(columns=columns)
  
  def get_formatted_matrix(self, input_lines, src_lang='en', **kwargs):
    res = np.array([[]]*len(input_lines))
    for encoder in self.encoders:
      x = encoder.transform(input_lines, src_lang=src_lang, output='matrix', **kwargs)
      if isinstance(x, list):
        x = np.array(x)
      elif not isinstance(x, np.ndarray):
        x = x.toarray()
      res = np.concatenate((res, x.reshape(len(input_lines),-1)), axis=1)
    res = preprocessing.normalize(res, norm='l2')
    return res.astype('float32')

  def size(self, **kwargs):
    return self.index.ntotal

  def add(self, input_lines, src_lang='en', faiss_index=None, **kwargs):
    if not input_lines:
      return self
    if faiss_index is not None:
      self.index = faiss_index
    starting_id = 0
    if self.index.ntotal:
      starting_id = 1+np.amax(faiss.vector_to_array(self.index.id_map), initial=0)
    ids = list(range(starting_id, starting_id+len(input_lines)))
    embeddings = kwargs.get("embeddings", None)
    if embeddings is None:
      embeddings = self.get_formatted_matrix(input_lines, src_lang=src_lang, **kwargs)
    self.index.add_with_ids(embeddings, np.array(ids))
    new_docs = pd.DataFrame({'document': input_lines, 'embedding': embeddings.tolist()})
    new_docs.index = ids
    self.corpus = pd.concat([self.corpus, new_docs])
    return self

  def remove(self, ids, faiss_index=None, **kwargs):
    if faiss_index is not None:
      self.index = faiss_index
    self.index.remove_ids(np.array(ids))
    self.corpus = self.corpus[~self.corpus.index.isin(ids)]

  def search(self, input_lines, src_lang='en', k=3, faiss_index=None, **kwargs):
    if faiss_index is not None:
      self.index = faiss_index
    if not self.index or not self.index.ntotal:
      warnings.warn('Empty results because no index found. Make sure to build your index before searching.')
      return []
    xq = self.get_formatted_matrix(input_lines, src_lang=src_lang, **kwargs)
    return self.index.search(xq, k)

  def retrieve(self, input_lines, k=3, **kwargs):
    distances, pred_ids = self.search(input_lines, k=k)
    return [self.corpus["document"].loc[[i for i in line_ids if i >= 0]].tolist() for line_ids in pred_ids]

  def transform(self, input_lines, src_lang='en', **kwargs):
    super().transform(input_lines, src_lang=src_lang, **kwargs)
    self.src_lang = src_lang
    d = self.get_formatted_matrix(["DUMMY"], src_lang=src_lang, **kwargs).shape[-1]
    self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
    if not input_lines:
      return self
    return self.add(input_lines, src_lang=src_lang, **kwargs)
