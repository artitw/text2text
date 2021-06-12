import torch

from text2text import Transformer
from text2text import Translator

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

class Answerer(Transformer):
  pretrained_answerer = "valhalla/longformer-base-4096-finetuned-squadv1"

  def __init__(self, **kwargs):
    self.__class__.pretrained_translator = kwargs.get('pretrained_translator')
    pretrained_answerer = kwargs.get('pretrained_answerer')
    if not pretrained_answerer:
      pretrained_answerer = self.__class__.pretrained_answerer
    self.__class__.tokenizer = AutoTokenizer.from_pretrained(pretrained_answerer)
    self.__class__.model = AutoModelForQuestionAnswering.from_pretrained(pretrained_answerer)

  def _translate_lines(self, input_lines, src_lang, tgt_lang):
    translator = getattr(self.__class__, "translator", Translator(pretrained_translator=self.__class__.pretrained_translator))
    self.__class__.translator = translator
    return translator.transform(input_lines, src_lang=src_lang, tgt_lang=tgt_lang)

  def _get_answers(self, input_lines):
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    num_examples = len(input_lines)
    encoded_inputs = tokenizer.batch_encode_plus(input_lines, padding=True, return_tensors="pt")
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]
    results = model(input_ids, attention_mask=attention_mask)
    ans_ids = [None] * num_examples
    for i in range(num_examples):
      max_startscore = torch.argmax(results["start_logits"][i])
      max_endscore = torch.argmax(results["end_logits"][i])
      ans_ids[i] = input_ids[i][max_startscore:max_endscore+1]
    answers = tokenizer.batch_decode(ans_ids, skip_special_tokens=True) 
    answers = [a.strip() for a in answers]
    return answers

  def transform(self, input_lines, src_lang='en', **kwargs):
    Transformer.transform(self, input_lines, src_lang, **kwargs)
    if src_lang != 'en':
      input_lines = self._translate_lines(input_lines, src_lang, 'en')

    input_lines = [line.split(" [SEP] ")[::-1] for line in input_lines]
    output_lines = self._get_answers(input_lines)

    if src_lang != 'en':
      output_lines = self._translate_lines(output_lines, src_lang='en', tgt_lang=src_lang)
          
    return output_lines