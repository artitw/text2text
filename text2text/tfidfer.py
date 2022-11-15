import numpy as np
import scipy.sparse as sp
import text2text as t2t

class Tfidfer(t2t.Counter):

  def _calculate_idf(self, token_counts):
    num_documents = len(token_counts)
    self.idf = {}
    for count in token_counts:
      counted = set()
      for tk in count:
        if tk not in self.idf:
          self.idf[tk] = 0
        if tk not in counted:
          self.idf[tk] += 1
          counted.add(tk)
    for tk in self.idf:
      self.idf[tk] = np.log(num_documents/(1+self.idf[tk]))+1

  def _get_vectors(self, token_counts):
    rows = []
    cols = []
    vals = []

    for i in range(len(token_counts)):
      num_tokens = sum(token_counts[i].values())
      for tk in token_counts[i]:
        token_counts[i][tk] /= num_tokens

      magnitude = 0
      for tk in token_counts[i]:
        token_counts[i][tk] *= getattr(self,'idf',{}).get(tk,1)
        magnitude += token_counts[i][tk]**2

      magnitude **= 0.5 
      for tk in token_counts[i]:
        token_counts[i][tk] /= magnitude
        rows.append(i)
        cols.append(tk)
        vals.append(token_counts[i][tk])

    if self.output == "matrix":
      x = sp.csr_matrix((vals,(rows,cols)))
      d = len(self.__class__.tokenizer.get_vocab())
      col_diff = d-x.shape[1]
      padding = sp.csr_matrix((x.shape[0], col_diff))
      token_counts = sp.hstack([x, padding], format="csr")
    return token_counts

  def transform(self, input_lines, src_lang='en', output='tokens', use_idf=True, **kwargs):
    self.output = output
    if output == "matrix":
      output = "ids"    
    token_counts = t2t.Counter.transform(self, input_lines, src_lang=src_lang, output=output, **kwargs)
    if use_idf:
      self._calculate_idf(token_counts)
    return self._get_vectors(token_counts)