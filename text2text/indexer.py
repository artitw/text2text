import text2text as t2t
import faiss
import numpy as np
import scipy.sparse as sp
import warnings

class Indexer(t2t.Searcher):

  def get_formatted_matrix(self, input_lines, src_lang='en', **kwargs):
    res = np.array([[]]*len(input_lines))
    for encoder in self.encoders:
      x = encoder().transform(input_lines, src_lang=src_lang, output='matrix', **kwargs)
      if not isinstance(x, np.ndarray):
        x = x.toarray()
      res = np.concatenate((res, x.reshape(len(input_lines),-1)), axis=1)
    return res.astype('float32')

  def size(self, **kwargs):
    return self.index.ntotal

  def add(self, input_lines, src_lang='en', ids=[], faiss_index=None, **kwargs):
    if faiss_index is not None:
      self.index = faiss_index
    if not ids:
      starting_id = 1+np.amax(faiss.vector_to_array(self.index.id_map), initial=0)
      ids = list(range(starting_id, starting_id+len(input_lines)))
    v = self.get_formatted_matrix(input_lines, src_lang=src_lang, **kwargs)
    self.index.add_with_ids(v, np.array(ids))
    return self

  def remove(self, ids, faiss_index=None, **kwargs):
    if faiss_index is not None:
      self.index = faiss_index
    self.index.remove_ids(np.array(ids))

  def search(self, input_lines, src_lang='en', k=3, faiss_index=None, **kwargs):
    if faiss_index is not None:
      self.index = faiss_index
    if not self.index or not self.index.ntotal:
      warnings.warn('Empty results because no index found. Make sure to build your index before searching.')
      return []
    xq = self.get_formatted_matrix(input_lines, src_lang=src_lang, **kwargs)
    return self.index.search(xq, k)

  def transform(self, input_lines, src_lang='en', ids=[], encoders=[t2t.Tfidfer], **kwargs):
    self.encoders = encoders
    d = self.get_formatted_matrix(["DUMMY"], src_lang=src_lang, **kwargs).shape[-1]
    print(f"Creating index with {d} dimensions.")
    self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(d))
    if not input_lines:
      return self
    return self.add(input_lines, src_lang=src_lang, ids=ids, **kwargs)
