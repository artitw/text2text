import text2text as t2t

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
        "yes", "yeah", "yep", "sure", "absolutely", "definitely", 
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
    sqlite_path = kwargs.get("sqlite_path", None)
    if sqlite_path:
      conn = sqlite3.connect(sqlite_path)
      query = f"SELECT document, embedding FROM {RAG_TABLE_NAME}"
      self.records = pd.read_sql_query(query, conn)
      conn.close()
      self.records["embedding"] = self.records["embedding"].apply(lambda x: pickle.loads(x))
      self.index = t2t.Indexer().transform([], encoders=[t2t.Vectorizer()])
      self.index.add(
        self.records["document"].tolist(), 
        embeddings=np.vstack(self.records["embedding"])
      )
      return

    schema = kwargs.get("schema", None)
    texts = kwargs.get("texts", [])
    urls = kwargs.get("urls", [])
    input_lines = []
    for u in tqdm(urls, desc='Scrape URLs'):
      if is_valid_url(u):
        try:
          texts.append(get_cleaned_html(u))
        except Exception as e:
          warnings.warn(f"Skipping URL with errors: {u}")
      else:
        warnings.warn(f"Skipping invalid URL: {u}")
    
    if schema:
      column_names = schema.model_fields.keys()
      self.records = pd.DataFrame(columns=column_names)
      for t in tqdm(texts, desc='Schema extraction'):
        fields = ", ".join(column_names)
        prompt = f'Extract {fields} from the following text:\n\n{t}'
        res = t2t.Assistant.chat_completion(self, [{"role": "user",  "content": prompt}], schema=schema)
        new_row = pd.DataFrame([vars(res)])
        self.records = pd.concat([self.records, new_row], ignore_index=True)
        res = "\n".join(f'{k}: {v}' for k,v in vars(res).items())
        input_lines.append(res)
    else:
      input_lines = texts
      self.records = pd.DataFrame({"text": texts})

    self.index = t2t.Indexer().transform(input_lines, encoders=[t2t.Vectorizer()])
    self.records = pd.concat([self.records, self.index.corpus], axis=1)
    self.records["embedding"] = self.records["embedding"].apply(lambda x: pickle.dumps(x))
    conn = sqlite3.connect("text2text.db")
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
    grounding_prompt = "Base your response on the following information:\n\n" + "\n- ".join(docs)
    messages[-1] = {"role": "user", "content": query + "\n\n" + grounding_prompt}
    return t2t.Assistant.chat_completion(self, messages=messages, stream=stream, schema=schema, **kwargs)
