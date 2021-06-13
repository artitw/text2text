import re
import glob
import math
from tqdm import tqdm
import numpy as np
import torch
import random
import requests, zipfile, io
import os

from .pytorch_pretrained_bert.tokenization import BertTokenizer
from .pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder

from .biunilm import seq2seq_loader

import text2text as t2t

class Abstractor(t2t.Transformer):
  pretrained_parameters = {}

  def _detokenize(self, tk_list):
    r_list = []
    for tk in tk_list:
      if tk.startswith('##') and len(r_list) > 0:
        r_list[-1] = r_list[-1] + tk[2:]
      else:
        r_list.append(tk)
    return r_list

  def _download_pretrained_model(self):
    pretrained_parameters = self.__class__.pretrained_parameters
    if os.path.isfile(pretrained_parameters["model_recover_path"]):
      print(f'{pretrained_parameters["model_recover_path"]} found in current directory.')
      return
    s = requests.session()
    file_id = pretrained_parameters["file_id"]
    r = s.get(f'https://docs.google.com/uc?export=download&id={file_id}')
    confirm_code = r.text.split("/uc?export=download&amp;confirm=")[1].split("&amp;id=")[0]
    r = s.get(f'https://docs.google.com/uc?export=download&confirm={confirm_code}&id={file_id}')
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

  def _get_token_id_set(self, s):
    r = None
    if s:
      w_list = []
      for w in s.split('|'):
        if w.startswith('[') and w.endswith(']'):
          w_list.append(w.upper())
        else:
          w_list.append(w)
      r = set(self.__class__.tokenizer.convert_tokens_to_ids(w_list))
    return r

  def __init__(self, **kwargs):
    self.__class__.pretrained_translator = kwargs.get('pretrained_translator')
    pretrained_parameters = self.__class__.pretrained_parameters
    if pretrained_parameters["max_tgt_length"] >= pretrained_parameters["max_seq_length"] - 2:
      raise ValueError("Maximum tgt length exceeds max seq length - 2.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    seed = self.__class__.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
      torch.cuda.manual_seed_all(seed)
    tokenizer = BertTokenizer.from_pretrained(pretrained_parameters["bert_model"], **pretrained_parameters)
    tokenizer.max_len = pretrained_parameters["max_seq_length"]

    pair_num_relation = 0
    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, pretrained_parameters["max_seq_length"], **pretrained_parameters))

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
      (1 if pretrained_parameters["s2s_add_segment"] else 0) if pretrained_parameters["new_segment_ids"] else 2
    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

    self.__class__.tokenizer = tokenizer

    forbid_ignore_set = self._get_token_id_set(pretrained_parameters["forbid_ignore_word"])
    not_predict_set = self._get_token_id_set(pretrained_parameters["not_predict_token"])

    self._download_pretrained_model()

    model_recover_path = glob.glob(pretrained_parameters["model_recover_path"].strip())[0]
    print(f"***** Recover model: {model_recover_path} *****")
    map_device = None
    if not torch.cuda.is_available():
      map_device='cpu'
    model_recover = torch.load(model_recover_path,map_location=map_device)
    pretrained_parameters["max_position_embeddings"] = pretrained_parameters["max_seq_length"]
    params = {k: v for k, v in pretrained_parameters.items() if k in BertForSeq2SeqDecoder.__dict__}
    model = BertForSeq2SeqDecoder.from_pretrained(pretrained_parameters["bert_model"], state_dict=model_recover, num_labels=cls_num_labels, num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id,
                                                  eos_id=eos_word_ids, sos_id=sos_word_id, forbid_ignore_set=forbid_ignore_set, not_predict_set=not_predict_set, **params)
    del model_recover

    if pretrained_parameters["fp16"]:
      model.half()
    model.to(device)
    if n_gpu > 1:
      model = torch.nn.DataParallel(model)
    torch.cuda.empty_cache()
    model.eval()

    self.__class__.device = device
    self.__class__.model = model
    self.__class__.bi_uni_pipeline = bi_uni_pipeline

  def _translate_lines(self, input_lines, src_lang, tgt_lang):
    translator = getattr(self.__class__, "translator", t2t.Translator(pretrained_translator=self.__class__.pretrained_translator))
    self.__class__.translator = translator
    return translator.transform(input_lines, src_lang=src_lang, tgt_lang=tgt_lang)

  def transform(self, input_lines, src_lang='en', **kwargs):
    t2t.Transformer.transform(self, input_lines, src_lang, **kwargs)
    if src_lang != 'en':
      input_lines = self._translate_lines(input_lines, src_lang, 'en')

    pretrained_parameters = self.__class__.pretrained_parameters
    tokenizer = self.__class__.tokenizer
    model = self.__class__.model
    bi_uni_pipeline = self.__class__.bi_uni_pipeline
    device = self.__class__.device

    max_src_length = pretrained_parameters["max_seq_length"] - 2 - pretrained_parameters["max_tgt_length"]
    input_lines = [tokenizer.tokenize(x)[:max_src_length] for x in input_lines]
    input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
    output_lines = [""] * len(input_lines)
    score_trace_list = [None] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / pretrained_parameters["batch_size"])
    next_i = 0
    with tqdm(total=total_batch) as pbar:
      while next_i < len(input_lines):
        _chunk = input_lines[next_i:next_i + pretrained_parameters["batch_size"]]
        buf_id = [x[0] for x in _chunk]
        buf = [x[1] for x in _chunk]
        next_i += pretrained_parameters["batch_size"]
        max_a_len = max([len(x) for x in buf])
        instances = []
        for instance in [(x, max_a_len) for x in buf]:
          for proc in bi_uni_pipeline:
            instances.append(proc(instance))
        with torch.no_grad():
          batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
          batch = [t.to(device) if t is not None else None for t in batch]
          input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
          traces = model(input_ids, token_type_ids,
                          position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
          output_ids = traces.tolist()
          for i in range(len(buf)):
            w_ids = output_ids[i]
            output_buf = tokenizer.convert_ids_to_tokens(w_ids)
            output_tokens = []
            for t in output_buf:
              if t in ("[SEP]", "[PAD]"):
                break
              output_tokens.append(t)
            output_sequence = ' '.join(self._detokenize(output_tokens))
            output_sequence = re.sub(r'\s([?.!"](?:\s|$))', r'\1', output_sequence)
            output_sequence = re.sub(r"\b\s+'\b", r"'", output_sequence)
            output_sequence = output_sequence.replace("[X_SEP]", "")
            output_lines[buf_id[i]] = output_sequence
        pbar.update(1)

    if src_lang != 'en':
      output_lines = self._translate_lines(output_lines, src_lang='en', tgt_lang=src_lang)
          
    return output_lines