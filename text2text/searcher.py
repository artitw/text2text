import numpy as np
from text2text import Tfidfer

class Searcher(Tfidfer):

  def predict(self, input_lines, queries, src_lang='en', **kwargs):
    tfidf = Tfidfer.predict(self, input_lines)
    if type(queries) != list:
      queries = [queries]
    queries = Tfidfer.predict(self, queries)
    scores = np.zeros((len(queries), len(tfidf)))
    for i, q in enumerate(queries):
      for j, d in enumerate(tfidf):
        for tk in q:
          scores[i,j] += q.get(tk,0)*d.get(tk,0)
    return scores