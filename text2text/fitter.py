import text2text as t2t
import torch
from tqdm import tqdm


class Fitter(t2t.Translator):

  def transform(self, input_lines, src_lang='en', tgt_lang='en', num_epochs=10, save_directory="model_dir", **kwargs):
    if tgt_lang not in self.__class__.LANGUAGES:
      raise ValueError(f'{tgt_lang} not found in {self.__class__.LANGUAGES}')
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    model.train()
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    split_lines = list(zip(*[line.split(" [TGT] ") for line in input_lines]))
    src_text, tgt_text = split_lines[0], split_lines[1]
    encoder_inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)
    with tokenizer.as_target_tokenizer():
        decoder_inputs = tokenizer(tgt_text, return_tensors="pt", padding=True, truncation=True).input_ids
    for epoch in tqdm(range(num_epochs)):
      results = model(**encoder_inputs, labels=decoder_inputs)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    return results