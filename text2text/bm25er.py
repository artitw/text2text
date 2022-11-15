import numpy as np
import scipy.sparse as sp
import text2text as t2t

class Bm25er(t2t.Tfidfer):

  def _get_vectors(self, token_counts):
    rows = []
    cols = []
    vals = []

    field_length_average = sum(
        sum(tk_cnts.values()) for tk_cnts in token_counts
    )/len(token_counts)

    for i in range(len(token_counts)):
      field_length_normalized = sum(token_counts[i].values())/field_length_average

      for tk in token_counts[i]:
        token_counts[i][tk] = getattr(self,'idf',{}).get(tk,1) * (
            token_counts[i][tk] * (self.k1+1) / (
                token_counts[i][tk] + self.k1 * (
                    1 - self.b + self.b * field_length_normalized
                )
            )
        )

      for tk in token_counts[i]:
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

  def transform(self, input_lines, src_lang='en', output='tokens', b=0.75, k1=1.0, **kwargs):
    self.output = output
    if output == "matrix":
      output = "ids"    
    self.b = b
    self.k1 = k1
    token_counts = t2t.Counter.transform(self, input_lines, src_lang=src_lang, output=output, **kwargs)
    self._calculate_idf(token_counts)
    return self._get_vectors(token_counts)