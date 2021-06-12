import random
import string
from text2text import Abstractor

class Questioner(Abstractor):
  pretrained_parameters = {
    "file_id": "1JN2wnkSRotwUnJ_Z-AbWwoPdP53Gcfsn",
    "fp16": False,
    "amp": False,
    "model_recover_path": "qg_model.bin",
    "max_seq_length": 512,
    "max_tgt_length": 48,
    "batch_size": 16,
    "search_beam_size": 1,
    "length_penalty": 0,
    "forbid_duplicate_ngrams": False,
    "forbid_ignore_word": None,
    "bert_model": "bert-large-cased",
    "ffn_type": 0,
    "num_qkv": 0,
    "seg_emb": False,
    "do_lower_case": False,
    "new_segment_ids": True,
    "min_len": None,
    "ngram_size": 3,
    "mode": "s2s",
    "s2s_special_token": False,
    "s2s_add_segment": False,
    "s2s_share_segment": False,
    "pos_shift": False,
    "not_predict_token": None,
  }

  def _get_random_answer(self, doc):
    unique_words = set(doc.lower().translate(str.maketrans('', '', string.punctuation)).split())
    answers = list(unique_words-self.__class__.STOP_WORDS)
    return random.choice(answers) if answers else random.choice(list(unique_words))

  def transform(self, input_lines, src_lang='en', **kwargs):
    if src_lang != 'en':
      input_lines = self._translate_lines(input_lines, src_lang, 'en')
    input_lines = [x + " [SEP] " + self._get_random_answer(x) if " [SEP] " not in x else x for x in input_lines]
    questions = Abstractor.transform(self, input_lines, src_lang='en', **kwargs)
    answers = [input.split(" [SEP] ")[1] for input in input_lines]
    if src_lang != 'en':
      questions = self._translate_lines(questions, 'en', src_lang)
      answers = self._translate_lines(answers, 'en', src_lang)
    return list(zip(questions, answers))