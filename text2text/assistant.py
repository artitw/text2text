import os
import sys
import torch
from hqq.core.quantize import BaseQuantizeConfig
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download
from .mixtral.build_model import OffloadConfig, QuantConfig, build_model
from transformers import TextStreamer

def _clean_output(input_prompt, output_text):
  input_prompt = input_prompt.replace('[INST]',' [INST] ').replace('  ',' ')
  output_text = output_text.replace('[INST]',' [INST] ').replace('  ',' ')
  return output_text.replace(input_prompt,"").replace('<s>',"").replace('</s>',"").strip()

class Assistant(object):
  def __init__(self, **kwargs):
    os.environ["LC_ALL"] = "en_US.UTF-8"
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia"
    os.environ["LIBRARY_PATH"] = "/usr/local/cuda/lib64/stubs"
    os.system("ldconfig /usr/lib64-nvidia")

    model_name = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"
    state_path = model_name
    repo_id = f"lavawolfiee/{model_name}"
    snapshot_download(repo_id=repo_id, local_dir=model_name)
    config = AutoConfig.from_pretrained(model_name)
    self.__class__.device = torch.device("cuda:0")

    offload_per_layer = 4 # Change to 5 if only 12 GB of GPU VRAM

    num_experts = config.num_local_experts

    offload_config = OffloadConfig(
        main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
        offload_size=config.num_hidden_layers * offload_per_layer,
        buffer_size=4,
        offload_per_layer=offload_per_layer,
    )

    attn_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        quant_zero=True,
        quant_scale=True,
    )
    attn_config["scale_quant_params"]["group_size"] = 256


    ffn_config = BaseQuantizeConfig(
        nbits=2,
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)


    self.__class__.model = build_model(
        device=self.__class__.device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )

    self.__class__.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.__class__.streamer = TextStreamer(self.__class__.tokenizer, skip_prompt=True, skip_special_tokens=True)
    self.__class__.cache = {}
  
  def chat_completion_tokens(self, messages):
    tokenizer = self.__class__.tokenizer
    device = self.__class__.device
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    return len(input_ids[0])

  def chat_completion(self, messages=[{"role": "user", "content": "hello"}], stream=True, **kwargs):
    tokenizer = self.__class__.tokenizer
    cache = self.__class__.cache
    device = self.__class__.device
    streamer = self.__class__.streamer
    model = self.__class__.model

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    past_input_string = tokenizer.apply_chat_template(messages[:-1], tokenize=False)
    past_key_values = cache.get(past_input_string, None)
    if past_key_values:
      seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
      attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)
    else:
      attention_mask = torch.ones_like(input_ids)

    results = model.generate(
      input_ids=input_ids,
      attention_mask=attention_mask,
      past_key_values=past_key_values,
      streamer=streamer if stream else None,
      do_sample=kwargs.get("do_sample", True),
      temperature=kwargs.get("temperature", 0.9),
      top_p=kwargs.get("top_p", 0.9),
      max_new_tokens=kwargs.get("max_new_tokens", 512),
      pad_token_id=tokenizer.eos_token_id,
      return_dict_in_generate=True,
      output_hidden_states=False,
    )

    output_string = tokenizer.batch_decode(**results)[0]
    input_string = tokenizer.apply_chat_template(messages, tokenize=False)
    messages.append({
      "role": "assistant",
      "content": _clean_output(input_string, output_string)
    })
    cache_string = tokenizer.apply_chat_template(messages, tokenize=False)
    self.__class__.cache[cache_string] = results["past_key_values"]

    return messages[-1]

  def transform(self, input_lines, src_lang='en', **kwargs):
    return self.chat_completion([{"role": "user", "content": input_lines}])["content"]

  completion = transform