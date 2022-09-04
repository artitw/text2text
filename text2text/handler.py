import text2text as t2t

class Handler(object):
  """
  Text Handler unified API for text transformers
  """

  EXPOSED_TRANSFORMERS = {
    "answer": t2t.Answerer,
    "bm25": t2t.Bm25er,
    "count": t2t.Counter,
    "index": t2t.Indexer,
    "fit": t2t.Fitter,
    "measure": t2t.Measurer,
    "question": t2t.Questioner,
    "search": t2t.Searcher,
    "summarize": t2t.Summarizer,
    "tfidf": t2t.Tfidfer,
    "tokenize": t2t.Tokenizer,
    "translate": t2t.Translator,
    "variate": t2t.Variator,
    "vectorize": t2t.Vectorizer,
    "identify":t2t.Identifier,
  }

  transformer_instances = {}

  def _transformer_handler(self, transformation, **kwargs):
    if transformation in self.__class__.transformer_instances:
      transformer = self.__class__.transformer_instances[transformation]
    else:
      transformer = self.__class__.EXPOSED_TRANSFORMERS[transformation]()
      self.__class__.transformer_instances[transformation] = transformer
    return transformer.transform(input_lines=self.input_lines, src_lang=self.src_lang, **kwargs)
    
  def __init__(self, input_lines=[], src_lang='en', **kwargs):
    self.input_lines = input_lines
    self.src_lang = src_lang
    for k in self.__class__.EXPOSED_TRANSFORMERS:
      handler = lambda x: lambda **kwargs: self._transformer_handler(transformation=x, **kwargs)
      handler = handler(k)
      setattr(self, k, handler)
