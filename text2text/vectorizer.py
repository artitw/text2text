import text2text as t2t

class Vectorizer(t2t.Assistant):

  def transform(self, input_lines, **kwargs):
    return self.embed(input_lines)