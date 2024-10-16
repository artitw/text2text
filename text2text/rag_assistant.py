import text2text as t2t

import os
import pickle
import sqlite3
import requests
import warnings
import urllib.parse

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from bs4 import BeautifulSoup

def get_cleaned_html(url):
  r = requests.get(url)
  soup = BeautifulSoup(r.text, 'html.parser')

  # Remove unwanted tags
  for script in soup(['script', 'style']):
      script.decompose()

  cleaned_text = soup.get_text(separator=' ', strip=True)

  return cleaned_text

def is_valid_url(url):
  try:
    result = urllib.parse.urlparse(url)
    return all([result.scheme, result.netloc])
  except Exception:
    return False

def is_affirmative(response):
    affirmative_keywords = [
        "y", "yes", "yeah", "yep", "sure", "absolutely", "definitely", 
        "certainly", "of course", "indeed", "affirmative", "correct", 
        "right", "exactly", "true", "positive"
    ]
    
    response_lower = response.lower()
    
    for keyword in affirmative_keywords:
        if keyword in response_lower:
            return True
            
    return False

RAG_TABLE_NAME = "rag_corpus_embeddings"

class RagAssistant(t2t.Assistant):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    schema = kwargs.get("schema", None)
    texts = kwargs.get("texts", [])
    urls = kwargs.get("urls", [])
    sqlite_path = kwargs.get("sqlite_path", None)
    self.index = t2t.CompositeIndexer()

    if urls:
      for u in tqdm(urls, desc='Scrape HTML'):
        if is_valid_url(u):
          try:
            texts.append(get_cleaned_html(u) + f"\nURL: {u}")
          except Exception as e:
            warnings.warn(f"Skipping URL with errors: {u}")
        else:
          warnings.warn(f"Skipping invalid URL: {u}")
    
    db_fields = {"document", "embedding"}
    if schema:
      column_names = schema.model_fields.keys()
      db_fields.update(column_names)
      fields = ", ".join(column_names)
      self.records = pd.DataFrame(columns=column_names)
      input_lines = []
      for t in tqdm(texts, desc='Extract Schema'):
        prompt = f'Extract {fields} from the following text:\n\n{t}'
        res = t2t.Assistant.chat_completion(self, [{"role": "user",  "content": prompt}], schema=schema)
        new_row = pd.DataFrame([vars(res)])
        self.records = pd.concat([self.records, new_row], ignore_index=True)
        res = "\n".join(f'{k}: {v}' for k,v in vars(res).items())
        input_lines.append(res)
      self.index.add(input_lines)
      self.records = pd.concat([self.records, self.index.corpus], axis=1)
    else:
      self.index.add(texts)
      self.records = self.index.corpus
    
    self.records["embedding"] = self.records["embedding"].apply(lambda x: pickle.dumps(x))

    if sqlite_path and os.path.exists(sqlite_path):
      conn = sqlite3.connect(sqlite_path)
      fields = ", ".join(db_fields)
      query = f"SELECT {fields} FROM {RAG_TABLE_NAME}"
      db_records = pd.read_sql_query(query, conn)
      db_records.dropna(subset=["document", "embedding"], inplace=True)
      conn.close()
      embeddings = db_records["embedding"].apply(lambda x: pickle.loads(x))
      embeddings = pd.DataFrame(embeddings.to_list())
      embeddings = [np.vstack(embeddings[col]) for col in embeddings.columns]
      self.index.add(
        db_records["document"].tolist(), 
        embeddings=embeddings
      )
      self.records = pd.concat([self.records, db_records], ignore_index=True)
    
    conn = sqlite3.connect(sqlite_path or "text2text.db")
    self.records.to_sql(RAG_TABLE_NAME, conn, if_exists='replace', index=False)
    conn.close()


  def chat_completion(self, messages=[{"role": "user", "content": "hello"}], stream=False, schema=None, **kwargs):
    k = kwargs.get("k", 3)
    query = messages[-1]["content"]
    question_check = f"Respond YES if this is a question; otherwise respond NO: {query}"
    question_check = [{"role": "user", "content": question_check}]
    response = t2t.Assistant.chat_completion(self, question_check)["message"]["content"]
    docs = []
    if is_affirmative(response):
      reword_prompt = f"Reword this question to be a demand: {query}"
      reword_prompt = [{"role": "user", "content": reword_prompt}]
      demand = t2t.Assistant.chat_completion(self, reword_prompt)["message"]["content"]
      docs = self.index.retrieve([demand], k=k)[0]
    else:
      docs = self.index.retrieve([query], k=k)[0]
    grounding_prompt = "Base your response on the following information:\n\n" + "\n\n".join(docs)
    messages[-1] = {"role": "user", "content": query + "\n\n" + grounding_prompt}
    return t2t.Assistant.chat_completion(self, messages=messages, stream=stream, schema=schema, **kwargs)
