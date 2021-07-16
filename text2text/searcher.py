import numpy as np
import scipy.sparse as sp
import text2text as t2t

class Searcher(t2t.Transformer):
  
  def transform(self, input_lines, queries, src_lang='en', vector_class=t2t.Tfidfer, index=None, **kwargs):
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang, **kwargs)
    search_object = vector_class()

    if index is None:
      index = search_object.transform(input_lines, src_lang='en', output="matrix")
    if type(queries) == str:
      queries = [queries]
    elif type(queries) != list:
      queries = list(queries)
    queries = search_object.transform(queries, src_lang='en', output="matrix", use_idf=False)
    
    if type(queries) is np.ndarray and type(index) is np.ndarray:
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
      raise TypeError("Invalid type for queries and/or index.")
    return scores