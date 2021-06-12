import numpy as np
from text2text import Tokenizer

class Counter(Tokenizer):

  def transform(self, input_lines, src_lang='en', output='tokens', **kwargs):
    token_ids = Tokenizer.transform(self, input_lines, src_lang=src_lang, output=output, **kwargs)
    token_counts = []
    for tokens in token_ids:
      token_counter = {}
      for tk in tokens:
        if tk not in token_counter:
          token_counter[tk] = 0
        token_counter[tk] += 1
      token_counts.append(token_counter)
    return token_counts