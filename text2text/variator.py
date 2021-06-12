from text2text import Transformer, Translator

class Variator(Translator):
  def transform(self, input_lines, src_lang='en', **kwargs):
    Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    output_lines = []
    src_lang = src_lang
    for tgt_lang in self.__class__.LANGUAGES:
      translated = self._translate(input_lines, src_lang=src_lang, tgt_lang=tgt_lang)
      output_lines += self._translate(translated, src_lang=tgt_lang, tgt_lang=src_lang)
    return output_lines 