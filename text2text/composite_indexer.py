import text2text as t2t

import pandas as pd

from collections import Counter

CORPUS_COLUMNS = ["document", "embedding"]

def aggregate_and_sort(x):
  # Flatten the list of lists
  flat_list = [item for sublist in x for item in sublist]
  # Count occurrences
  counts = Counter(flat_list)
  # Sort items by count (descending) and then by item (ascending)
  sorted_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
  # Return only the items in sorted order
  return [item[0] for item in sorted_items]

class CompositeIndexer(object):
  def __init__(self):
    index_sem = t2t.Indexer(encoders=[t2t.Vectorizer()]).transform([])
    index_syn = t2t.Indexer(encoders=[t2t.Tfidfer()]).transform([])
    self.indexes = [index_sem, index_syn]
    self.corpus = pd.DataFrame(columns=CORPUS_COLUMNS)

  def size(self, **kwargs):
    return len(self.corpus.index)

  def update_corpus(self):
    new_rows = pd.DataFrame(columns=CORPUS_COLUMNS)
    for index in self.indexes:
      new_rows = pd.concat([new_rows, index.corpus], ignore_index=False)
    self.corpus = new_rows.groupby(new_rows.index).agg({"document": "max", "embedding": list}).reset_index()

  def add(self, texts, **kwargs):
    embeddings = kwargs.get("embeddings", [None]*len(self.indexes))
    for i, index in enumerate(self.indexes):
      index.add(texts, embeddings=embeddings[i])
    self.update_corpus()
    return self
    
  def remove(self, ids, faiss_index=None, **kwargs):
    for index in self.indexes:
      index.remove(ids)
    self.update_corpus()

  def retrieve(self, input_lines, **kwargs):
    df = pd.DataFrame({"document": []})
    for index in self.indexes:
      res = index.retrieve(input_lines, **kwargs)
      df2 = pd.DataFrame({"document": res})
      df = pd.concat([df, df2], axis=0)
    df = df.groupby(df.index).agg(aggregate_and_sort)
    df.reset_index(drop=True, inplace=True)
    return df["document"].tolist()

  def transform(self, input_lines, src_lang='en', **kwargs):
    if not input_lines:
      return self
    return self.add(input_lines, src_lang=src_lang, **kwargs)