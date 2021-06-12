import numpy as np
from text2text import Translator, Transformer

class Vectorizer(Translator):

  def transform(self, input_lines, src_lang='en', **kwargs):
    Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    tokenizer.src_lang = src_lang
    encoded_inputs = tokenizer(input_lines, padding=True, return_tensors="pt")
    outputs = model.forward(**encoded_inputs, decoder_input_ids=encoded_inputs["input_ids"])
    last_layer_states = outputs.encoder_last_hidden_state.detach().numpy()
    input_ids = encoded_inputs["input_ids"].detach().numpy()
    non_paddings = input_ids!=1
    non_paddings = non_paddings.astype(int)
    non_paddings = np.repeat(non_paddings, last_layer_states.shape[-1], axis=1)
    non_paddings = non_paddings.reshape(last_layer_states.shape)
    x = np.average(last_layer_states, axis=1, weights=non_paddings)
    x /= np.repeat(np.linalg.norm(x, axis=1),x.shape[-1]).reshape(x.shape)
    return list(x)