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
    last_layer_states = outputs.encoder_last_hidden_state.detach().numpy()
    input_ids = encoded_inputs["input_ids"].detach().numpy()
    non_paddings = input_ids!=1
    non_paddings = non_paddings.astype(int)
    non_paddings = np.repeat(non_paddings, last_layer_states.shape[-1], axis=1)
    non_paddings = non_paddings.reshape(last_layer_states.shape)
    embeddings = np.average(last_layer_states, axis=1, weights=non_paddings)
    return embeddings