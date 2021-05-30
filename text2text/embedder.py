import numpy as np
from text2text import Translator, Transformer

class Embedder(Translator):

  def predict(self, input_lines, src_lang='en', **kwargs):
    Transformer.predict(self, input_lines, src_lang=src_lang, **kwargs)
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    tokenizer.src_lang = src_lang
    encoded_inputs = tokenizer(input_lines, padding=True, return_tensors="pt")
    outputs = model.forward(**encoded_inputs, decoder_input_ids=encoded_inputs["input_ids"])
    return np.mean(outputs.encoder_last_hidden_state.detach().numpy(), 1)