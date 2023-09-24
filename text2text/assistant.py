import logging
import pandas as pd
import text2text as t2t
from transformers import AutoTokenizer, logging
from auto_gptq import AutoGPTQForCausalLM

logging.set_verbosity(logging.CRITICAL)

def _clean_output(input_prompt, output_text):
  return output_text.replace('<s>',"").replace('</s>',"").replace(input_prompt, "").strip()

class Assistant(t2t.Transformer):

  def __init__(self, **kwargs):
    model_name_or_path = kwargs.get("model_name_or_path", "TheBloke/vicuna-13B-v1.5-16K-GPTQ")

    self.__class__.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    self.__class__.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
      use_safetensors=True,
      trust_remote_code=False,
      device="cuda:0",
      use_triton=False,
      quantize_config=None
    )

  def completion_preprocess(self, input_lines, retriever=None, **kwargs):
    df = pd.DataFrame({"input_line": input_lines})
    if retriever:
      k = kwargs.get('k', 1)
      df["knowledge"] = retriever.retrieve(df["input_line"].str.lower().tolist(), k=k)
      df["input_line"] = df["knowledge"].apply(' '.join) + " - " + df["input_line"]
    df["input_line"] = "USER: " + df["input_line"] + "\nASSISTANT:"
    return df

  def completion_tokens(self, input_lines):
    df = self.completion_preprocess(input_lines)
    tok = self.__class__.tokenizer
    input_ids = tok(df["input_line"].tolist(), return_tensors="pt", padding=True).input_ids
    return [len(x) for x in input_ids]

  def transform(self, input_lines, retriever=None, **kwargs):
    df = self.completion_preprocess(input_lines, retriever, **kwargs)
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.95)
    top_k = kwargs.get('top_k', 0)
    repetition_penalty = kwargs.get('repetition_penalty', 1.15)
    max_new_tokens = kwargs.get('max_new_tokens', 512)
    tok = self.__class__.tokenizer
    m = self.__class__.model

    input_ids = tok(df["input_line"].tolist(), return_tensors="pt", padding=True).input_ids
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

    df["output_line"] = tok.batch_decode(m.generate(**generate_kwargs)) 
    df["output_line"] = df.apply(lambda row: _clean_output(row["input_line"], row["output_line"]), axis=1)

    return df["output_line"].tolist()

  completion = transform

  def chat_completion_preprocess(self, messages):
    chat_history = [f'{line["role"].upper()}: {line["content"]}' for line in messages]
    chat_history.append("ASSISTANT: ")
    input_prompt = "\n".join(chat_history)
    return input_prompt

  def chat_completion_tokens(self, messages):
    input_prompt = self.chat_completion_preprocess(messages)
    tok = self.__class__.tokenizer
    input_ids = tok([input_prompt], return_tensors="pt", padding=True).input_ids[0]
    return len(input_ids)

  def chat_completion(self, messages, **kwargs):
    input_prompt = self.chat_completion_preprocess(messages)
      
    temperature = kwargs.get('temperature', 0.7)
    top_p = kwargs.get('top_p', 0.95)
    top_k = kwargs.get('top_k', 0)
    repetition_penalty = kwargs.get('repetition_penalty', 1.15)
    max_new_tokens = kwargs.get('max_new_tokens', 512)
    tok = self.__class__.tokenizer
    m = self.__class__.model

    input_ids = tok([input_prompt], return_tensors="pt", padding=True).input_ids
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
    
    results = tok.batch_decode(m.generate(**generate_kwargs))[0]
    return {
      "role": "assistant",
      "content": _clean_output(input_prompt, results)
    }
