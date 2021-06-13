import numpy as np
import scipy.sparse as sp
from text2text import Transformer

class Searcher(Transformer):

  def __init__(self, **kwargs):
    pretrained_translator = kwargs.get('pretrained_translator')
    if not pretrained_translator:
      pretrained_translator = self.__class__.pretrained_translator
    self.__class__.pretrained_translator = pretrained_translator

  def transform(self, input_lines, queries, src_lang='en', class_name="Tfidfer", index=None, **kwargs):
    Transformer.transform(self, input_lines, src_lang, **kwargs)
    t2t_module = __import__("text2text")
    search_class = getattr(t2t_module, class_name)
    search_object = search_class(pretrained_translator=self.__class__.pretrained_translator)

    if index is None:
      index = search_object.transform(input_lines, src_lang='en', output="matrix")
    if type(queries) != list:
      queries = [queries]
    queries = search_object.transform(queries, src_lang='en', output="matrix", use_idf=False)
    
    if type(queries[0]) is dict and type(index[0]) is dict:
      scores = np.zeros((len(queries), len(index)))
      for i, q in enumerate(queries):
        for j, d in enumerate(index):
            for tk in q:
              scores[i,j] += q.get(tk,0)*d.get(tk,0)
    elif type(queries[0]) is np.ndarray and type(index[0]) is np.ndarray:
      scores = np.matmul(queries,np.transpose(index))
    elif type(queries) is sp.csr_matrix and type(index) is sp.csr_matrix:
      col_diff = queries.shape[1]-index.shape[1]
      if col_diff > 0:
        padding = sp.csr_matrix((index.shape[0], abs(col_diff)))
        index = sp.hstack([index, padding], format="csr")
      elif col_diff < 0:
        padding = sp.csr_matrix((queries.shape[0], abs(col_diff)))
        queries = sp.hstack([queries,padding], format="csr")
      scores = queries.dot(index.transpose())
    else:
      TypeError("Invalid type for queries and/or index.")
    return scores