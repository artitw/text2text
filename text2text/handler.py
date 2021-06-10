from text2text import Answerer, Vectorizer, Measurer, Questioner, Summarizer, Tokenizer, Translator, Variator

class Handler(object):
  """
  Text Handler unified API for text transformers
  """

  EXPOSED_TRANSFORMERS = {
    "answer": Answerer,
    "vectorize": Vectorizer,
    "measure": Measurer,
    "question": Questioner,
    "summarize": Summarizer,
    "tokenize": Tokenizer,
    "translate": Translator,
    "variate": Variator,
  }

  def _transformer_handler(self, transformation, tgt_lang):
    transformer = self.__class__.transformer_instances.get(transformation, self.__class__.EXPOSED_TRANSFORMERS[transformation](pretrained_translator=self.__class__.pretrained_translator))
    self.__class__.transformer_instances[transformation] = transformer
    return transformer.predict(input_lines=self.input_lines, src_lang=self.src_lang, tgt_lang=tgt_lang)
    
  def __init__(self, input_lines, src_lang='en', **kwargs):
    self.input_lines = input_lines
    self.src_lang = src_lang
    self.__class__.pretrained_translator = kwargs.get("pretrained_translator")
    for k in self.__class__.EXPOSED_TRANSFORMERS:
      handler = lambda x: lambda tgt_lang="en", **kwargs: self._transformer_handler(transformation=x, tgt_lang=tgt_lang)
      handler = handler(k)
      setattr(self, k, handler)
    self.__class__.transformer_instances = {}