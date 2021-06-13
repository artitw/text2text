import text2text as t2t

def levenshtein_distance(s1, s2):
  prev_row = list(range(1,len(s2)+1)) + [0]
  for i in range(len(s1)):
    cur_row = [0]*len(s2) + [i+1]
    for j in range(len(s2)):
      del_cost = prev_row[j] + 1
      add_cost = cur_row[j-1] + 1
      sub_cost = prev_row[j-1] + (s1[i]!=s2[j])
      cur_row[j] = min(del_cost, add_cost, sub_cost)
    prev_row = cur_row
  return cur_row[len(s2)-1]

class Measurer(t2t.Tokenizer):

  def transform(self, input_lines, src_lang='en', metric='levenshtein_distance', **kwargs):
    t2t.Transformer.transform(self, input_lines, src_lang=src_lang, **kwargs)
    tokenizer = self.__class__.tokenizer
    tokenizer.src_lang = src_lang
    input_lines = list(zip(*[l.split(" [SEP] ") for l in input_lines]))
    encoded_inputs_a = tokenizer(input_lines[0], add_special_tokens=False)
    encoded_inputs_b = tokenizer(input_lines[1], add_special_tokens=False)
    output_lines = []
    for i,input_ids_a in enumerate(encoded_inputs_a["input_ids"]):
      input_ids_b = encoded_inputs_b["input_ids"][i]
      output_lines.append(levenshtein_distance(input_ids_a, input_ids_b))
    return output_lines