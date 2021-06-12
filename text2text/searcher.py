import numpy as np
from text2text import Tfidfer

class Searcher(Tfidfer):

  def transform(self, input_lines, queries, src_lang='en', search_index=None, **kwargs):
    if search_index is None:
      search_index = Tfidfer.transform(self, input_lines)
    if type(queries) != list:
      queries = [queries]
    queries = Tfidfer.transform(self, queries, use_idf=False)
    scores = np.zeros((len(queries), len(search_index)))
    for i, q in enumerate(queries):
      for j, d in enumerate(search_index):
        for tk in q:
          scores[i,j] += q.get(tk,0)*d.get(tk,0)
    return scores