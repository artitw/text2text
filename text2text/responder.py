import torch
import text2text as t2t

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

  def transform(self, input_lines, src_lang='en', **kwargs):
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang, **kwargs)
    if src_lang != 'en':
      input_lines = self._translate_lines(input_lines, src_lang, 'en')

    output_lines = self._get_responses(input_lines)

    if src_lang != 'en':
      output_lines = self._translate_lines(output_lines, src_lang='en', tgt_lang=src_lang)
          
    return output_lines
