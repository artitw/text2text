import torch
import numpy as np
import text2text as t2t

class Vectorizer(t2t.Translator):

  def transform(self, input_lines, src_lang='en', output_dimension=1, **kwargs):
    input_lines = t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    device = self.__class__.device
    tokenizer.src_lang = src_lang
    encoder_inputs = tokenizer(input_lines, padding=True, return_tensors="pt").to(device)
    outputs = model.forward(**encoder_inputs, decoder_input_ids=torch.zeros(len(input_lines),1,dtype=int).to(device))
    last_layer_states = outputs.encoder_last_hidden_state.cpu().detach().numpy()
    if output_dimension==1:
      x = np.mean(last_layer_states, axis=1)
      x /= np.linalg.norm(x, axis=1).reshape(x.shape[0],-1)
      return x
    return last_layer_states