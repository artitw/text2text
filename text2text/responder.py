import text2text as t2t
import pandas as pd
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Responder(t2t.Answerer):
  pretrained_model = "microsoft/GODEL-v1_1-large-seq2seq"

  def __init__(self, **kwargs):
    pretrained_model = kwargs.get('pretrained_model')
    if not pretrained_model:
      pretrained_model = self.__class__.pretrained_model
    self.__class__.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    self.__class__.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

  def _get_responses(self, input_lines):
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model

    inputs = tokenizer(input_lines, return_tensors="pt", padding=True)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=128, min_length=8, top_p=0.9, do_sample=True,
    )

    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

  def transform(self, input_lines, src_lang='en', knowledge_base=None, **kwargs):
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang, **kwargs)
    df = pd.DataFrame({"input_lines": input_lines})
    df.loc[len(df)] = "[CONTEXT][KNOWLEDGE]"
    cols = ["instruction", "context", "knowledge"]
    df[cols] = df["input_lines"].str.split(r"\[CONTEXT\]|\[KNOWLEDGE\]", expand=True)
    df.drop(df.tail(1).index, inplace=True)
    df.fillna("", inplace=True)
    df[cols] = df[cols].apply(lambda x: x.str.strip())

    if src_lang != 'en':
      df["instruction"] = self._translate_lines(df["instruction"].tolist(), src_lang, 'en')
      df["context"] = self._translate_lines(df["context"].tolist(), src_lang, 'en')
      df["knowledge"] = self._translate_lines(df["knowledge"].tolist(), src_lang, 'en')

    if knowledge_base:
      corpus, index = knowledge_base
      df["knowledge"] = df.apply(lambda row: corpus[index.search([row["context"].lower()], k=1)[1][0][0]] if not row["knowledge"] else row["knowledge"], axis=1)

    df["input_lines"] = df["instruction"] + " [CONTEXT] " + df["context"]
    df["input_lines"] = df.apply(lambda row: row["input_lines"] + " [KNOWLEDGE] " + row["knowledge"] if row["knowledge"] else row["input_lines"], axis=1)
    
    output_lines = self._get_responses(df["input_lines"].tolist())

    if src_lang != 'en':
      output_lines = self._translate_lines(output_lines, src_lang='en', tgt_lang=src_lang)
          
    return output_lines
