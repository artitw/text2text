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

class Summarizer(t2t.Abstractor):
  pretrained_parameters = {
    "file_id": "1RyJxShxC9tDYVAyZwUwqkSoQ3l5DfjuE",
    "fp16": True,
    "amp": True,
    "model_recover_path": "cnndm_model.bin",
    "max_seq_length": 768,
    "max_tgt_length": 128,
    "batch_size": 64,
    "search_beam_size": 5,
    "length_penalty": 0,
    "forbid_duplicate_ngrams": True,
    "forbid_ignore_word": ".|[X_SEP]",
    "bert_model": "bert-large-cased",
    "ffn_type": 0,
    "num_qkv": 0,
    "seg_emb": False,
    "do_lower_case": False,
    "new_segment_ids": True,
    "min_len": None,
    "ngram_size": 3,
    "mode": "s2s",
    "s2s_special_token": False,
    "s2s_add_segment": False,
    "s2s_share_segment": False,
    "pos_shift": False,
    "not_predict_token": None,
  }
