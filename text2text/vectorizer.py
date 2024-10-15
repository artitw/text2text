import text2text as t2t
from tqdm.auto import tqdm

class Vectorizer(t2t.Assistant):

  def transform(self, input_lines, **kwargs):
    sentencize = kwargs.get("sentencize", True)
    if sentencize and input_lines != ["DUMMY"]:
      sentences = []
      for text in tqdm(input_lines, desc='Summarize'):
        if len(text) > 100:
          prompt = f'Summarize the following text to a single sentence:\n\n{text}'
          result = self.chat_completion([{"role": "user",  "content": prompt}])
          sentences.append(result["message"]["content"])
        else:
          sentences.append(text)
      return self.embed(sentences)
    return self.embed(input_lines)