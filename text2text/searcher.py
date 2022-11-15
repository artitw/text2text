import numpy as np
import scipy.sparse as sp
import text2text as t2t

class Searcher(t2t.Transformer):
  
  def transform(self, input_lines, queries, src_lang='en', vector_class=t2t.Bm25er, index=None, **kwargs):
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
      scores = queries.dot(index.transpose())
    else:
      raise TypeError("Invalid type for queries and/or index.")
    return scores