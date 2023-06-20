import text2text as t2t
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

class Assistant(t2t.Transformer):

  def __init__(self, **kwargs):
    model_name_or_path = "TheBloke/vicuna-13b-v1.3-GPTQ"
    model_basename = "vicuna-13b-v1.3-GPTQ-4bit-128g.no-act.order"

    self.__class__.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    self.__class__.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
      model_basename=model_basename,
      use_safetensors=True,
      trust_remote_code=False,
      device="cuda:0",
      use_triton=False,
      quantize_config=None
    )

  def transform(self, input_lines, src_lang='en', **kwargs):
    input_lines = [f'''USER: {prompt}\nASSISTANT:''' for prompt in input_lines]
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.9)
    top_k = kwargs.get('top_k', 0)
    repetition_penalty = kwargs.get('repetition_penalty', 1.1)
    max_new_tokens = kwargs.get('max_new_tokens', 512)
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

    output_lines = tok.batch_decode(m.generate(**generate_kwargs)) 

    for i in range(len(input_lines)):
      output_lines[i] = output_lines[i].replace('<s>',"").replace('</s>',"").replace(input_lines[i], "").strip()
    
    return output_lines
