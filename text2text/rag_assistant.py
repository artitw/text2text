import text2text as t2t

import requests
import warnings
import urllib.parse

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

class RagAssistant(t2t.Assistant):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
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
      for t in tqdm(texts, desc='Schema extraction'):
        fields = ", ".join(schema.model_fields.keys())
        prompt = f'Extract {fields} from the following text:\n\n{t}'
        res = t2t.Assistant.chat_completion(self, [{"role": "user",  "content": prompt}], schema=schema)
        res = "\n".join(f'{k}: {v}' for k,v in vars(res).items())
        input_lines.append(res)
    else:
      input_lines = texts

    self.index = t2t.Indexer().transform(input_lines, encoders=[t2t.Vectorizer()])

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
