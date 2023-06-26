import torch
import text2text as t2t
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator(t2t.Transformer):

  def __init__(self, **kwargs):
    pretrained_translator = self.__class__.PRETRAINED_TRANSLATOR
    self.__class__.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_translator, device_map="auto", load_in_8bit=True)
    self.__class__.tokenizer = AutoTokenizer.from_pretrained(pretrained_translator)

  def _translate(self, input_lines, src_lang='en', **kwargs):
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    tokenizer.src_lang = src_lang
    if 'tgt_lang' not in kwargs:
      raise ValueError('tgt_lang not specified')
    tgt_lang = kwargs.get('tgt_lang')
    if src_lang==tgt_lang:
      return input_lines
    if tgt_lang not in self.__class__.LANGUAGES:
      raise ValueError(f'{tgt_lang} not found in {self.__class__.LANGUAGES}')
    encoded_inputs = tokenizer(input_lines, padding=True, truncation=True, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    tgt_token_id = tokenizer.lang_code_to_id[tgt_lang]
    generated_tokens = model.generate(**encoded_inputs, forced_bos_token_id=tgt_token_id)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True) 

  def transform(self, input_lines, src_lang='en', **kwargs):
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    return self._translate(input_lines, src_lang=src_lang, **kwargs)