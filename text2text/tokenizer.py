from text2text import Transformer
from transformers import AutoTokenizer

class Tokenizer(Transformer):

  def __init__(self, **kwargs):
    pretrained_translator = kwargs.get('pretrained_translator')
    if not pretrained_translator:
      pretrained_translator = self.__class__.pretrained_translator
    self.__class__.pretrained_translator = pretrained_translator
    self.__class__.tokenizer = AutoTokenizer.from_pretrained(pretrained_translator)

  def predict(self, input_lines, src_lang='en', **kwargs):
    Transformer.predict(self, input_lines, src_lang=src_lang, **kwargs)
    tokenizer = self.__class__.tokenizer
    tokenizer.src_lang = src_lang
    encoded_inputs = tokenizer(input_lines, add_special_tokens=False)
    output_lines = []
    for input_ids in encoded_inputs["input_ids"]:
      output_lines.append(tokenizer.convert_ids_to_tokens(input_ids))
    return output_lines