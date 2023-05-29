import torch
import text2text as t2t
from peft import PeftModel    
from transformers import AutoModelForCausalLM, LlamaTokenizer

class Assistant(t2t.Transformer):

  def __init__(self, **kwargs):
    model_name = "decapoda-research/llama-7b-hf"
    adapters_name = 'timdettmers/guanaco-7b'
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    m = PeftModel.from_pretrained(m, adapters_name)
    self.__class__.model = m.merge_and_unload()
    self.__class__.tokenizer = LlamaTokenizer.from_pretrained(model_name)
    self.__class__.tokenizer.bos_token_id = 1

  def transform(self, input_lines, src_lang='en', **kwargs):
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.9)
    top_k = kwargs.get('top_k', 0)
    repetition_penalty = kwargs.get('repetition_penalty', 1.1)
    max_new_tokens = kwargs.get('max_new_tokens', 1536)
    tok = self.__class__.tokenizer
    m = self.__class__.model
    input_ids = tok(input_lines, return_tensors="pt").input_ids
    input_ids = input_ids.to(m.device)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    return tok.batch_decode(m.generate(**generate_kwargs), skip_special_tokens=True) 
    
