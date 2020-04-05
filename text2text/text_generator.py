import glob
import math
from tqdm import tqdm
import numpy as np
import torch
import random
import requests, zipfile, io
import os

from .pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from .pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder

from .biunilm import seq2seq_loader

STOP_WORDS = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",]

def detokenize(tk_list):
  r_list = []
  for tk in tk_list:
    if tk.startswith('##') and len(r_list) > 0:
      r_list[-1] = r_list[-1] + tk[2:]
    else:
      r_list.append(tk)
  return r_list

def ascii_print(text):
  text = text.encode("ascii", "ignore")
  print(text)

PRETRAINED_PARAMS = {
  "question": {
    "file_id": "1JN2wnkSRotwUnJ_Z-AbWwoPdP53Gcfsn",
    "fp16": False,
    "amp": False,
    "model_recover_path": "qg_model.bin",
    "max_seq_length": 512,
    "max_tgt_length": 48,
    "batch_size": 16,
    "beam_size": 1,
    "length_penalty": 0,
    "forbid_duplicate_ngrams": False,
    "forbid_ignore_word": None
  },
  "summary": {
    "file_id": "1RyJxShxC9tDYVAyZwUwqkSoQ3l5DfjuE",
    "fp16": True,
    "amp": True,
    "model_recover_path": "cnndm_model.bin",
    "max_seq_length": 768,
    "max_tgt_length": 128,
    "batch_size": 64,
    "beam_size": 5,
    "length_penalty": 0,
    "forbid_duplicate_ngrams": True,
    "forbid_ignore_word": ".|[X_SEP]"
  }
}

class TextGenerator(object):

  def __init__(self, output_type="question", **kwargs):
    self.output_type = output_type
    self.bert_model = "bert-large-cased"
    self.ffn_type = 0
    self.num_qkv = 0
    self.seg_emb = False
    self.split = "test"
    self.seed = 123
    self.do_lower_case = False
    self.new_segment_ids = True
    self.new_pos_ids = False
    self.min_len = None
    self.ngram_size = 3
    self.mode = "s2s"
    self.s2s_special_token = False
    self.s2s_add_segment = False
    self.s2s_share_segment = False
    self.pos_shift = False
    self.not_predict_token = None
    self.__dict__.update(PRETRAINED_PARAMS[output_type])
    self.__dict__.update(kwargs)

    if output_type not in ["question","summary"]:
      raise ValueError(f'{output_type} unacceptable for output_type. Choose either "question" or "summary".')

    if self.max_tgt_length >= self.max_seq_length - 2:
      raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    if n_gpu > 0:
      torch.cuda.manual_seed_all(self.seed)

    self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)

    self.tokenizer.max_len = self.max_seq_length

    pair_num_relation = 0
    self.bi_uni_pipeline = []
    self.bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, self.max_seq_length, max_tgt_length=self.max_tgt_length, new_segment_ids=self.new_segment_ids,
                                                                    mode="s2s", num_qkv=self.num_qkv, s2s_special_token=self.s2s_special_token, s2s_add_segment=self.s2s_add_segment, s2s_share_segment=self.s2s_share_segment, pos_shift=self.pos_shift))

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 + \
      (1 if self.s2s_add_segment else 0) if self.new_segment_ids else 2
    mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

    forbid_ignore_set = self._get_token_id_set(self.forbid_ignore_word)
    not_predict_set = self._get_token_id_set(self.not_predict_token)

    self.download_pretrained_model()

    for model_recover_path in glob.glob(self.model_recover_path.strip()):
      print("***** Recover model: %s *****", model_recover_path)
      model_recover = torch.load(model_recover_path)
      self.model = BertForSeq2SeqDecoder.from_pretrained(self.bert_model, state_dict=model_recover, num_labels=cls_num_labels, num_rel=pair_num_relation, type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=self.beam_size,
                                                    length_penalty=self.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=self.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set, not_predict_set=not_predict_set, ngram_size=self.ngram_size, min_len=self.min_len, mode=self.mode, max_position_embeddings=self.max_seq_length, ffn_type=self.ffn_type, num_qkv=self.num_qkv, seg_emb=self.seg_emb, pos_shift=self.pos_shift)
      del model_recover

      if self.fp16:
        self.model.half()
      self.model.to(self.device)
      if n_gpu > 1:
        self.model = torch.nn.DataParallel(self.model)

      torch.cuda.empty_cache()
      self.model.eval()

  def download_pretrained_model(self):
    if os.path.isfile(self.model_recover_path):
      print(f"{self.model_recover_path} found in current directory.")
      return
    s = requests.session()
    file_id = self.file_id
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
      r = set(self.tokenizer.convert_tokens_to_ids(w_list))
    return r

  def _get_answer_tokens(self, tkns):
    words = detokenize(tkns)
    answers = []
    for w in words:
      if len(w) > 1:
        if w.lower() not in STOP_WORDS:
          answers.append(w)
    return self.tokenizer.tokenize(random.choice(answers) if answers else words[0])

  def predict(self, input_lines, tokenized_input=False):
    data_tokenizer = WhitespaceTokenizer() if tokenized_input else self.tokenizer
    max_src_length = self.max_seq_length - 2 - self.max_tgt_length
    input_lines = [data_tokenizer.tokenize(x)[:max_src_length] for x in input_lines]

    if self.output_type=="question":
      input_lines = [x + ["[SEP]"] + self._get_answer_tokens(x) if "[SEP]" not in x else x for x in input_lines]

    input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
    output_lines = [""] * len(input_lines)
    score_trace_list = [None] * len(input_lines)
    total_batch = math.ceil(len(input_lines) / self.batch_size)
    next_i = 0
    with tqdm(total=total_batch) as pbar:
      while next_i < len(input_lines):
        _chunk = input_lines[next_i:next_i + self.batch_size]
        buf_id = [x[0] for x in _chunk]
        buf = [x[1] for x in _chunk]
        next_i += self.batch_size
        max_a_len = max([len(x) for x in buf])
        instances = []
        for instance in [(x, max_a_len) for x in buf]:
          for proc in self.bi_uni_pipeline:
            instances.append(proc(instance))
        with torch.no_grad():
          batch = seq2seq_loader.batch_list_to_batch_tensors(instances)
          batch = [t.to(self.device) if t is not None else None for t in batch]
          input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
          traces = self.model(input_ids, token_type_ids,
                          position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
          if self.beam_size > 1:
            traces = {k: v.tolist() for k, v in traces.items()}
            output_ids = traces['pred_seq']
          else:
            output_ids = traces.tolist()
          for i in range(len(buf)):
            w_ids = output_ids[i]
            output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
            output_tokens = []
            for t in output_buf:
              if t in ("[SEP]", "[PAD]"):
                break
              output_tokens.append(t)
            output_sequence = ' '.join(detokenize(output_tokens))
            output_sequence = output_sequence.replace(" ' ", "'").replace(" ?", "?")
            if self.output_type=="question":
              ans_idx = buf[i].index("[SEP]")
              corresponding_answer = ' '.join(detokenize(buf[i][ans_idx+1:]))
              output_lines[buf_id[i]] = (output_sequence, corresponding_answer)
            else:
              output_lines[buf_id[i]] = output_sequence
        pbar.update(1)
          
    return output_lines