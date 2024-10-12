# Text2Text Language Modeling Toolkit

<details>
  <summary>Overview</summary>

* [Colab Notebooks](https://github.com/artitw/text2text#colab-notebooks)
* [Installation Requirements](https://github.com/artitw/text2text#installation-requirements)
* [Quick Start Guide](https://github.com/artitw/text2text#api-quick-start-guide)
* [Class Diagram](https://github.com/artitw/text2text#class-diagram)
* [Languages Available](https://github.com/artitw/text2text#languages-available)
* [Requirements & Installation](https://github.com/artitw/text2text#requirements-and-installation)
* [Examples](https://github.com/artitw/text2text#examples)
  * [Assistant](https://github.com/artitw/text2text#assistant)
  * [Tokenization](https://github.com/artitw/text2text#tokenization)
  * [Embedding](https://github.com/artitw/text2text#embedding--vectorization)
  * [TF-IDF](https://github.com/artitw/text2text#tf-idf)
  * [BM25](https://github.com/artitw/text2text#bm25)
  * [Index](https://github.com/artitw/text2text#index)
  * [Distance](https://github.com/artitw/text2text#levenshtein-sub-word-edit-distance)
  * [Translation](https://github.com/artitw/text2text#translation)
  * [Data Augmentation](https://github.com/artitw/text2text#data-augmentation--back-translation)
  * [Finetuning](https://github.com/artitw/text2text#training--finetuning)
  * [Identification](https://github.com/artitw/text2text#identification)
  * [Web Server](https://github.com/artitw/text2text#serving)
* [Questions?](https://github.com/artitw/text2text#questions)
* [Citation](https://github.com/artitw/text2text#citation)
* [Contributing](https://github.com/artitw/text2text#contributing)
* [Code of Conduct](https://github.com/artitw/text2text#code-of-conduct)

</details>

## Colab Notebooks
* Assistant (free private ChatGPT LLM alternative) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K6Kk80w9vjFZ7PL9dPRgVuOPuaWcY4ae?usp=sharing)
* STF-IDF multilingual search [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RaWj5SqWvyC2SsCTGg8IAVcl9G5hOB50?usp=sharing)
* All examples [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LE_ifTpOGO5QJCKNQYtZe6c_tjbwnulR)

## Installation Requirements
```
pip install -qq -U text2text
```
* [Examples](#examples) run with <16 GB RAM on free [Colab GPUs](https://colab.research.google.com/drive/1LE_ifTpOGO5QJCKNQYtZe6c_tjbwnulR).

## Quick Start Guide
Functionality | Invocation | Result
:------------: | :-------------: | :-------------:
Module Importing | `import text2text as t2t` | Libraries imported
[Assistant](https://github.com/artitw/text2text#assistant) | `t2t.Assistant().transform("Describe Text2Text in a few words: ")` | `['Text2Text is an AI-powered text generation tool that creates coherent and continuous text based on prompts.']`
[Language Model Setting](https://github.com/artitw/text2text#byot-bring-your-own-translator) | `t2t.Transformer.PRETRAINED_TRANSLATOR = "facebook/m2m100_418M"` | Change from default
[Tokenization](https://github.com/artitw/text2text#tokenization) | `t2t.Tokenizer().transform(["Hello, World!"])` | `[['▁Hello', ',', '▁World', '!']]`
[Embedding](https://github.com/artitw/text2text#embedding--vectorization) | `t2t.Vectorizer().transform(["Hello, World!"])` | `[[0.18745188, 0.05658336, ..., 0.6332584 , 0.43805206]]`
[TF-IDF](https://github.com/artitw/text2text#tf-idf) | `t2t.Tfidfer().transform(["Hello, World!"])` | `[{'!': 0.5, ',': 0.5, '▁Hello': 0.5, '▁World': 0.5}]`
[BM25](https://github.com/artitw/text2text#bm25) | `t2t.Bm25er().transform(["Hello, World!"])` | `[{'!': 0.3068528194400547, ',': 0.3068528194400547, '▁Hello': 0.3068528194400547, '▁World': 0.3068528194400547}]`
[Indexer](https://github.com/artitw/text2text#index) | `index = t2t.Indexer().transform(["Hello, World!"])` | Index object for information retrieval
[Translation](https://github.com/artitw/text2text#translation) | `t2t.Translater().transform(["Hello, World!"], src_lang="en, tgt_lang="zh")` | `['你好,世界!']`
[Data Augmentation](https://github.com/artitw/text2text#data-augmentation--back-translation) | `t2t.Variator().transform(["Hello, World!"], src_lang="en)` | `['Hello the world!', 'Welcome to the world.', 'Hello to the world!',...`
[Distance](https://github.com/artitw/text2text#levenshtein-sub-word-edit-distance) | `t2t.Measurer().transform(["Hello, World! [SEP] Hello, what?"])` | `[2]`
[Training/Finetuning](https://github.com/artitw/text2text#training--finetuning) | `t2t.Fitter().transform(["Hello, World! [TGT] Hello, what?"])` | Finetuned model saved
[Identification](https://github.com/artitw/text2text#identification) | `t2t.Identifier().transform(["Aj keď sa Buzz Aldrin stal až „druhým človekom“..."])` | `['sk', 'Slovak']`
[Web Server](https://github.com/artitw/text2text#serving) | `t2t.Serve(host='0.0.0.0', port=80)` | Web server started on host and port

## Examples

### Assistant
- Free private open source alternative to commercial LLMs.
- Commercial LLMs are costly, collect your data, impose quotas and rate limits that hinder development.
- Run at no cost on Google Colab free tier, so you don't even need your own device.

```
import text2text as t2t
asst = t2t.Assistant()

# Streaming example
chat_history = [
    {"role": "user",  "content": "Hi"},
    {"role": "assistant", "content": "Hello, how are you?"},
    {"role": "user", "content": "What should I do today?"}
]

result = asst.chat_completion(chat_history, stream=True) #{'role': 'assistant', 'content': '1. Make a list of things to be grateful for.\n2. Go outside and take a walk in nature.\n3. Practice mindfulness meditation.\n4. Connect with a loved one or friend.\n5. Do something kind for someone else.\n6. Engage in a creative activity like drawing or writing.\n7. Read an uplifting book or listen to motivational podcasts.'}
for chunk in result:
  print(chunk['message']['content'], end='', flush=True)

# Running conversation
messages = []
while True:
  user_input = input("User: ")
  print()
  messages.append({"role": "user", "content": user_input})
  print("Assistant: ")
  result = asst.chat_completion(messages, stream=False)
  print(result["message"]["content"])
  messages.append(result["message"])
  print()

# Schema for structured output
from pydantic import BaseModel

class Song(BaseModel):
    name: str
    artist: str

result = asst.chat_completion([
    {"role": "user",  "content": "What is Britney Spears's best song?"}
], schema=Song)
# Song(name='Toxic', artist='Britney Spears')

# Embeddings
asst.embed(["hello, world!", "this will be embedded"])
```

### Tokenization
```
t2t.Tokenizer().transform([
  "Let's go hiking tomorrow",
  "안녕하세요.",
  "돼지꿈을 꾸세요~~"
])

# Sub-word tokens
[['▁Let', "'", 's', '▁go', '▁hik', 'ing', '▁tom', 'orrow'],
 ['▁안녕', '하세요', '.'],
 ['▁', '돼', '지', '꿈', '을', '▁꾸', '세요', '~~']]
```

### Embedding / Vectorization
```
t2t.Vectorizer().transform([
  "Let's go hiking tomorrow",
  "안녕하세요.",
  "돼지꿈을 꾸세요~~"
])

# Embeddings
[[-0.00352954,  0.0260059 ,  0.00407429, ..., -0.04830331,
  -0.02540749, -0.00924972],
  [ 0.00043362,  0.00249816,  0.01755436, ...,  0.04451273,
    0.05118701,  0.01895813],
  [-0.03563676, -0.04856304,  0.00518898, ..., -0.00311068,
    0.00071953, -0.00216325]]
```

### TF-IDF
```
t2t.Tfidfer().transform([
  "Let's go hiking tomorrow",
  "안녕하세요.",
  "돼지꿈을 꾸세요~~"
])

# TF-IDF values
[{'!': 0.22360679774997894,
  "'": 0.44721359549995787,
  ',': 0.22360679774997894,
  'ing': 0.22360679774997894,
  'orrow': 0.22360679774997894,
  's': 0.44721359549995787,
  '▁Let': 0.22360679774997894,
  '▁go': 0.44721359549995787,
  '▁hik': 0.22360679774997894,
  '▁let': 0.22360679774997894,
  '▁tom': 0.22360679774997894},
 {'.': 0.5773502691896258,
  '▁안녕': 0.5773502691896258,
  '하세요': 0.5773502691896258},
 {'~~': 0.3535533905932738,
  '▁': 0.3535533905932738,
  '▁꾸': 0.3535533905932738,
  '꿈': 0.3535533905932738,
  '돼': 0.3535533905932738,
  '세요': 0.3535533905932738,
  '을': 0.3535533905932738,
  '지': 0.3535533905932738}]
```

### BM25
```
t2t.Bm25er().transform([
  "Let's go hiking tomorrow",
  "안녕하세요.",
  "돼지꿈을 꾸세요~~"
])

# BM25 values
[{"'": 1.2792257271403649,
  'ing': 1.2792257271403649,
  'orrow': 1.2792257271403649,
  's': 1.2792257271403649,
  '▁Let': 1.2792257271403649,
  '▁go': 1.2792257271403649,
  '▁hik': 1.2792257271403649,
  '▁tom': 1.2792257271403649},
 {'.': 1.751071282233123, '▁안녕': 1.751071282233123, '하세요': 1.751071282233123},
 {'~~': 1.2792257271403649,
  '▁': 1.2792257271403649,
  '▁꾸': 1.2792257271403649,
  '꿈': 1.2792257271403649,
  '돼': 1.2792257271403649,
  '세요': 1.2792257271403649,
  '을': 1.2792257271403649,
  '지': 1.2792257271403649}]
```

### Index
[![STF-IDF Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RaWj5SqWvyC2SsCTGg8IAVcl9G5hOB50?usp=sharing)
```
index = t2t.Indexer().transform([
  "Let's go hiking tomorrow, let's go!",
  "안녕하세요.",
  "돼지꿈을 꾸세요~~",
])

index.retrieve(["돼지"], k=1) #[['"돼지꿈을 꾸세요~~"']]

# Add documents
index.add(["Hello, World! 你好,世界!"])

# Remove by ids
index.remove([2]) #Removes "돼지꿈을 꾸세요~~"

# Retrieve k results per query sorted by distance
index.retrieve(["你好, World"], k=3)
```
To learn more, see [STF-IDF](https://arxiv.org/abs/2209.14281).

### Levenshtein Sub-word Edit Distance
```
t2t.Measurer().transform([
  "Hello, World! [SEP] Hello, what?",
  "안녕하세요. [SEP] 돼지꿈을 꾸세요~~"
], metric="levenshtein_distance")

# Distances
 [2, 8]
```

### Translation
```
# Sample texts
article_en = 'The Secretary-General of the United Nations says there is no military solution in Syria.'

notre_dame_str = "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student - run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one - page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut."

bacteria_str = "Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats."

bio_str = "Biology is the science that studies life. What exactly is life? This may sound like a silly question with an obvious answer, but it is not easy to define life. For example, a branch of biology called virology studies viruses, which exhibit some of the characteristics of living entities but lack others. It turns out that although viruses can attack living organisms, cause diseases, and even reproduce, they do not meet the criteria that biologists use to define life."

t2t.Translator().transform([
  article_en,
  notre_dame_str,
  bacteria_str,
  bio_str
], src_lang='en', tgt_lang='zh')

# Translations
['联合国秘书长说,叙利亚没有军事解决方案。',
 '与大多数其他大学一样,Notre Dame的学生运行的新闻媒体渠道的数量。九个学生 - 运行的渠道包括三份报纸,两台广播电视台,以及几本杂志和杂志。 开始作为一个一页的杂志在1876年9月,该杂志的Schoolistic发行了每月两次,并声称是美国最古老的连续的大学新闻出版物,和其他杂志,TheJuggler,每年发行两次,并专注于学生文学和艺术作品。 多姆年刊每年发行。 报纸有不同的出版利益,与The Observer发表每日,主要报道大学和其他新闻,并由学生从Notre Dame和圣玛丽的学院。 与Scholastic和The Dome不同,The Observer是一个独立的公众作品,但没有教师顾',
 '细菌是生物细胞的一种类型. 它们构成一个大范围的亲生微生物. 通常几微米长,细菌有许多形状,从球到杖和螺旋。 细菌是地球上出现的第一个生命形式之一,并且存在于其大多数栖息地。',
 '生物学是研究生命的科学. 究竟什么是生命? 这可能听起来像一个愚蠢的问题,有一个显而易见的答案,但它并不容易定义生命. 例如,一个名为病毒学的生物学分支研究病毒,这些病毒表现出一些活体的特征,但缺乏其他。']
```

#### BYOT: Bring Your Own Translator
 * The default translator requires no more than 16GB of memory.
 * You can specify other pretrained translators at your own risk.
 * Make sure src_lang and tgt_lang codes conform to that model.

<details>
  <summary>BYOT examples</summary>

```
t2t.Transformer.PRETRAINED_TRANSLATOR = "facebook/mbart-large-50-many-to-many-mmt"
t2t.Transformer.LANGUAGES = {
  'af_ZA': 'Afrikaans',
  'ar_AR': 'Arabic',
  'az_AZ': 'Azerbaijani',
  'bn_IN': 'Bengali',
  'cs_CZ': 'Czech',
  'de_DE': 'German',
  'en_XX': 'English',
  'es_XX': 'Spanish',
  'et_EE': 'Estonian',
  'fa_IR': 'Persian',
  'fi_FI': 'Finnish',
  'fr_XX': 'French',
  'gl_ES': 'Galician',
  'gu_IN': 'Gujarati',
  'he_IL': 'Hebrew',
  'hi_IN': 'Hindi',
  'hr_HR': 'Croatian',
  'id_ID': 'Indonesian',
  'it_IT': 'Italian',
  'ja_XX': 'Japanese',
  'ka_GE': 'Georgian',
  'kk_KZ': 'Kazakh',
  'km_KH': 'Khmer',
  'ko_KR': 'Korean',
  'lt_LT': 'Lithuanian',
  'lv_LV': 'Latvian',
  'mk_MK': 'Macedonian',
  'ml_IN': 'Malayalam',
  'mn_MN': 'Mongolian',
  'mr_IN': 'Marathi',
  'my_MM': 'Burmese',
  'ne_NP': 'Nepali',
  'nl_XX': 'Dutch',
  'pl_PL': 'Polish',
  'ps_AF': 'Pashto',
  'pt_XX': 'Portuguese',
  'ro_RO': 'Romanian',
  'ru_RU': 'Russian',
  'si_LK': 'Sinhala',
  'sl_SI': 'Slovene',
  'sv_SE': 'Swedish',
  'sw_KE': 'Swahili',
  'ta_IN': 'Tamil',
  'te_IN': 'Telugu',
  'th_TH': 'Thai',
  'tl_XX': 'Tagalog',
  'tr_TR': 'Turkish',
  'uk_UA': 'Ukrainian',
  'ur_PK': 'Urdu',
  'vi_VN': 'Vietnamese',
  'xh_ZA': 'Xhosa',
  'zh_CN': 'Chinese'
}
t2t.Translator().transform(
  ["I would like to go hiking tomorrow."],
  src_lang="en_XX",
  tgt_lang='zh_CN'
)
['我想明天去徒步旅行。']
```

</details>

### Data Augmentation / Back-Translation
Back-translations useful for augmenting training data
```
t2t.Variator().transform([bacteria_str], src_lang='en')
```

<details>
  <summary>Show results</summary>

```
# Variations
['Bacteria are a kind of biological cell. They form a large domain of prokaryotic micro-organisms. Typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to borders and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of its habitats.',
 'Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles Chronicles',
 'Bacteria are a type of biological cell, forming a large scale of procariotic microbials, usually a few micrometers in length, and have many shapes, ranging from bodies to roots and heads, and bacteria were among the first forms of life that appeared on Earth, and they are present in most of their environments.',
 'The bacteria are a type of biological cell. They constitute a great domain of procariotic microorganisms. Typically a few micrometres of length, the bacteria have a series of shapes, variing from spheres to rays and spirals.',
 'Bacteria are a type of biological cell. They form a large range of procaryotic microorganisms. Typically, at a length of several micrometers, bacteria have a lot of shapes, between spheres and spirals. Bacteria have been among the first forms of life on Earth, and are in most of its habitats.',
 'Bacteria type biological cell. They make up a large domain of procariotic microorganisms. typically several micrometers long, bacteria have a number of formats, from spheres to roses and spirals. bacteria were among the first life formats to appear on Earth, and are present in most of its habitats.',
 'Bacteria are types of biological cells. They make up a great home of procaryotical microorganisms. Usually with a few micrometers of length, bacteria have several shapes that are located from spheres to species and spirals. Bacteria were from the first forms of life that appeared on Earth, and are found in many of their animals.',
 'Bacteria are a type of biological cell. They make up a large range of procariotic microorganisms. Usually several micrometers in length, bacteria have a number of shapes ranging from spheres to stairs and spirals. Bacteria are among the first forms of life that appear on Earth, and are available in most of its habitats.',
 'Bacteria are a type of organic cell. They create a larger area of procreatic microorganisms. usually a few micrometers long, the bacteria have a number of sizes, the streets and spirals from the surfers. the bacteria were in the first life sizes appeared on Earth, and present in most of its reality.',
 'Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats.',
 'The bacteria are a type of biological cell.They make a large domain of procarytic microorganisms.Typically a few micrometers long, the bacteria have a number of forms, from spheres to roots and spirals.The bacteria were among the first life forms that appear on Earth, and are present in most of their habitats.',
 'Bacteria are a type of biological cell. they constitute a great domain of prokarotic microorganisms. Typically a few micrometers of length, the bacteria have a series of shapes, which vary from spheres to wheels and spirals. The bacteria were among the first forms of life that appear on Earth, and are present in most of their habitats.',
 'The bacteria are a type of biological cell. The bacteria are a different domain of prokaryotic microorganisms. The bacteria are a form, from spheres to wraps and spirals. The bacteria is one of the first life forms of appearance on Earth, and is one of the habitats.',
 'Bacteria are a type of biological cells. they form a large area of prokaryotic microorganisms. Typically several micrometers in length, bacteria have several shapes, from spheres to roots and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of its residential spaces.',
 'Bacteria is a type of biological cell. It is fat in the fat in the prokaryotic microorganisms. The fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat in the fat.',
 'Bacteria are a type of biological cell. They form a large domain of prokaryotic microorganisms. Typically a couple of micrometers in length, bacteria have a variety of forms, ranging from spheres to layers and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cell. They form a large range of prokaryotic microorganisms. Normally a few micrometers in length, bacteria have a number of forms, from spheres to roots and spirals. Bacteria were among the first forms of life to appear on Earth, and are present in most of their habitats.',
 'Bacteria are a kind of organic cells. they make up a large range of prehyroid microorganisms. Usually a few micrometer length, bacteria have a number of forms, ranging from spheres to roots and spherals. The bacteria were among the first forms of life that appeared on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to roots and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cell. they constitute a great domain of procariotic microorganisms. Typically a few micrometers of length, bacteria have a series of shapes, which vary from spheres to roots and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of their habitats.',
 'Bacteria are a type of biological cells. They form a large area of prokarotic microorganisms. usually a few micrometers long, bacteria have a number of forms that range from spheres to roots and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of their habitats.',
 'Bacteria are a type of biological cells.They make up a large range of procaryotic microorganisms.The bacteria usually have a variety of shapes, ranging from branches to branches and spirals.The bacteria were among the first vital forms that appear on earth and are present in most of its habitats.',
 'Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cell. They form a large area of prokaryotic microorganisms. Typically a few micrometers length, bacteria have several forms, ranging from spheres to roots and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of its habitat environments.',
 'Bacteria are a type of biological cell. They constitute a large range of procaryotic microorganisms. Usually, a few micrometers long, the bacteria have a number of shapes, ranging from spheres to roots and spirals. The bacteria were among the first forms of life to appear on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cells. They set up a large domain of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a variety of forms, ranging from spheres to rectum and spirals. Bacteria are among the first forms of life that come to the Earth, and are in the most of their habitats.',
 'Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced, Inexperienced',
 'Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria are a number of shapes, ranges from spheres to roads and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cell. They constitute a great domain of procariotic microorganisms. Typically a few micrometers of length, the bacteria have a series of shapes, which vary from spheres to roots and spirals. The bacteria were among the first forms of life that appeared on Earth, and are present in most of their habitats.',
 'On the other hand, it is important to note that in the past few decades, there is a significant increase in the number of people living in the country, and there is a significant increase in the number of people living in the country, and there is a significant increase in the number of people living in the country, and there is a significant increase in the number of people living in the country.',
 'Bacteria is a type of biological cell. It has a big domain and prokaryotic microorganisms. It has a micrometres at the time, bacteria has in a number of shapes, ranging from spheres to rods and spirals. Bacteria is in the first and life shapes to appear on Earth, and is in the best and best forms.',
 'Bacteria are a type of biological cells.They constitute a large area of procreatic microorganisms.Generally a few micrometers long, bacteria have a number of forms, formed from literature to waves and spirals.Bacteria were among the first forms of life to appear on Earth, and they exist in most aspects.',
 'Bacteria are a type of organic cells. They create a large range of procariotic microorganisms. Usually in a few micrometers length, bacteria have many shapes, ranging from spare to wire and wire. Bacteria were one of the first forms of life that appeared on Earth, and are present in most of their population.',
 'Bacteria are a type of biological cell. they make a large domain of procarytic microorganisms. usually a few micrometers in length, bacteria have a number of forms, from spheres to roots and spirals. bacteria were among the first forms of life that appear on Earth, and are present in most of their habitats.',
 'Bacteria are a kind of biological cell. It represents a large domain of pro-cariotic microorganisms. Typically in some micrometres of length, bacteria have a lot of training, from the spheres of their hands and spirals. Bacteria were one of the first training of life appearing in the world, and are present in the majority of their habitats.',
 'Bacteria are a kind of biological cell. These form a large range of procarytic microorganisms. Generally, some micrometer long bacteria are in a variety of shapes, from spheres to fiber and spirals. Bacteria were among the first forms of life to appear on Earth and are present in most of their habitats.',
 "Related Topics: Faith  Faith as a Function of Prayer  Faith as Gift of God  Faith, Living  Fasting  Loyalty  Prayer  Prayer as Contact with God  Prayer as Conversation  Prayer as Fellowship  Prayer's Purpose  Prayer, Effectiveness in  Prayer, Power of  Praying Always  Praying at All Times  Praying without Ceasing  Relationship with God  Trust  Unbelief  Weak Prayer Life",
 'Bacteria are a type of biological cell. they form a large domain of prokaryotic microorganisms. usually a few micrometers long, bacteria have several shapes, from spheres to roots and spirals. Bacteria are among the first forms of life that appear on Earth, and present in most of its habitats.',
 'Bacteria is a type of biological cell. It provides a big domain of prokaryotic microorganisms. It provides some micrometres to spread, bacteria has many types, such as spheres and rods and spirals. Bacteria is the primary high-type species to be found on Earth, and is covered by many of its habitats.',
 'The bacteria is a type of biological cell. The bacteria is a type of bacteria that can be found to be found to be found to be found to be found to be found to be found to be found to be found to be found to be found to be found to be found to be found to be found to be found.',
 'Bacteria are a type of biological cells. They are a large area of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a number of species, ranging from areas to wounds and spirals. Bacteria were among the first biological phenomena to appear on the earth, and are located in the most biological phenomena.',
 'Bacteria are a type of biological cell. they constitute a wide domain of procariotic microorganisms. usually a few micrometers of length, bacteria have a number of shapes, ranging from spheres to roots and spirals. bacteria are among the first forms of life that appear on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biocells They make up a wide range of procariotic microorganisms, usually in a number of micrometres of length, bacteria have a variety of shapes from spheres to wings and spirals Bacteria are one of the first forms of life that appear on Earth and are present in most of their habitat.',
 'Bacteria is a type of biological cell.It is a large domain of prokaryotic microorganisms.Al a few micrometers in length, bacteria have a number of shapes, spread from spheres to grass and spiral.Bacteria are among the first forms of life to be displayed on Earth, and are present in most habitats.',
 'Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats. (cancer, transport, transport, transport, transport)',
 'The bacteria have biological cells. They are spread by procariotic microorganisms. The substance has several micrometers, the bacteria have many forms, spheres and spirals.',
 'Tag is a type of tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag tag',
 'It is true that there is a great deal of interest in the world, and there is a great deal of interest in the world, and there is a great deal of interest in the world, and there is a great deal of interest in the world, and there is a great deal of interest in the world, and there is a great deal of interest in the world, and there is a great deal of interest in the world.',
 'Bacteria are a type of biological cells.They make up a large range of procariotic microorganisms.In general, a few meters long, the bacteria have several forms from spheres to roots and spheres.The bacteria were one of the first forms of life to appear on Earth, and exist in most ecosystems.',
 'Bacteria are a type of biological cell. They form a large domain of prokaryotic microorganisms. Typically a few micrometers of length, bacteria have a number of forms, from spheres to spheres and spirals. Bacteria were between the first forms of life and appeared on Earth, and are in the majority of habitats.',
 'Bacteria is a type of biological cell. They are in a big domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria has a number of shapes, ranging from spheres to roads and spirals. Bacteria was among the first life forms to appear on Earth, and are present in most of its habitats.',
 'Bacteria of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot of makolinhot.',
 'Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Types of micrometers in the latter, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria were among the first forms of life to appear on Earth, and are renowned in more than its kind.',
 'Bacteria are a type of biological cells. They make up a large area of procarytic microorganisms. Usually a few micrometers long, bacteria have several forms ranging from spheres to branches and spirals. Bacteria were among the first forms of life that appeared on Earth, and are in many of its habitats.',
 'Bacteria are a type of biological cells. they form a large range of prokaryotic microorganisms. usually a few micrometers long, bacteria have several shapes, from spheres to roots and spirals. bacteria were among the first forms of life that appeared on Earth, and are located in the majority of its habitat.',
 'The bacteria is a type of biological cell. It creates a large scale of prokaryotic microorganisms. Usually small, the bacteria have a variety of forms, to spheres and spirals. The bacteria is one of the first forms of life found in the earth, and is present in most of its habitats.',
 'Bacteria are a type of biological cells. they represent a large area of procariotic microorganisms. Usually several micrometers in length, bacteria have a large number of shapes, ranging from spheres to genes and spirals. Bacteria were among the first forms of life to appear on Earth, and are present in most of its habitats.',
 'bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. bacteria were among the first life forms to appear on Earth, and are present in most of its habitats. bacteria are among the first life forms to appear on Earth, and are present in most of its habitats. bacteria are among the first life forms to appear on Earth, and are present in most of its habitats. bacteria are among the first life forms to appear on Earth.',
 'Bacteria are a biological pattern. They are a multi-facet of procariotic microorganisms. Moreover, with a micrometer, there are different types of bacteria shapes and spirals. The bacteria was one of the first life patterns in the world and it has a large pattern.',
 'Bacteria are types of biological cells. They become a large domain of procariotic microbes. Usually some micrometers are long, there are many shapes of bacteria, from spera to spira and spila. Bacteria were the first life shapes to appear on Earth, and are present in most of them.',
 'Bacteria are a kind of biological cell.They form a large domain of prokaryotic microorganisms.Overall a few micrometers in length, bacteria have several shapes, from spheres to roots and spirals.Bacteria are among the first forms of life that appear on Earth, and present in most of its habitats.',
 'bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. bacteria were among the first life shapes to appear on Earth, and are present in most of its habitats. The bacteria kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind kind',
 '“That’s what I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do, I’m going to do.”',
 'Bacteria are a kind of biological cell. They form a large domain of prokaryotic microorganisms. Typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to rocks and spirals. Bacteria were among the first forms of life that appear on Earth, and are present in most of the habitats.',
 'Bacteria are a type of biological cell. They form a large area of prokaryotic microorganisms. Usually a few micrometers in length, bacteria have a variety of forms, from spheres to rows and spirals. Bacteria were among the first life forms that appeared on Earth, and are present in most of its habitats.',
 'small small small small small small small small small small small small small small small small small small small small small small small small small small small small small small small small',
 'Bacteria are a type of biological cell, they constitute a large range of procariotic microorganisms, typically micrometers long, bacteria are a variety of shapes, from spheres to wheels and spirals, and bacteria are among the first forms of life that appeared on Earth, and are present in most of its habitats.',
 'bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria have a number of shapes, ranging from spheres to roads and spirals. bacteria were among the first life forms to appear on Earth, and are present in most of its habitats.',
 'Bacteria is a kind of bacteria. They have a large area of prokaryotic microorganisms. Usually some micrometers are a kind of bacteria, from fresh to color and from fresh. Bacteria as first bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria as bacteria.',
 'Bacteria are a type of biological cells. they make up a large area of prokaryotic microorganisms. usually a few micrometers long, bacteria have a number of shapes, from spheres to corners and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cell. They form a large range of procariotic microorganisms. In the usual length of several micrometers, bacteria have many shapes, from shapes to colors and spirals. Bacteria are in the first life shapes that appear on earth, and are present in many beauty.',
 'Bacteria are a type of biological cell. they constitute a great domain of procarotic microorganisms. typically, some micrometers of length, the bacteria have a number of shapes, ranging from spheres to roots and spirals. the bacteria were among the first forms of life that appeared on Earth, and are present in most of their habitats.',
 'Bacteria are a type of biological cell. they constitute a large range of procariotic microorganisms. usually, at a few micrometers of length, bacteria have a number of forms, from spheres to rows and spirals. bacteria have been among the first forms of life that appeared on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cells. They make up a large domain of procariotic microorganisms. As a rule, several micrometers in length, bacteria have a number of forms, from spheres to births and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of their fields.',
 'Bacteria is a section of biological negative. These are a large section of procariotic microorganisms. Generally, in the length of several micrometers, bacteria are in different forms, in sperm and sperm. Bacteria were in the earlier life forms of appearance on earth, and most of them are in life.',
 'bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. typically a few micrometers in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. bacteria were among the first life forms to appear on Earth, and are present in most of its habitats. bacteria are among the first life forms to appear on Earth, and are present in most of its habitats. bacteria are among the first life forms to appear on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cells. they form a large area of prokaryotic microorganisms. Typically several micrometers in length, bacteria have several shapes, ranging from spheres to tribes and spirals. Bacteria were among the first forms of life that appeared on Earth, and are present in most of its living places.',
 'Bacteria are a type of biological cell. they make up a large area of procarotic microorganisms. Usually a few micrometres of length, bacteria have a number of forms, from the sphere to the wings and spirals. Bacteria were among the first life forms that occur on Earth, and are present in most of their habitats.',
 'Bakteer is from mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid to mid',
 'Bacteria are a type of biological cells.They make up a large range of prokaryotic microorganisms.Typically some micrometers in length, bacteria have a number of forms, dispersing from the sphere to the root and spirals.Bacteria were among the first forms of life that appeared on Earth, and are present in most of its habitats.',
 'Bacteria are a type of biological cell.They make up a large domain of procariotic microorganisms.Generally a few micrometers in length, bacteria have a range of shapes, ranging from spheres to genes and spirals.Bacteria were among the first forms of life that appear on Earth, and are present in most of its habitats.',
 'Bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria, bacteria',
 'Bacteria are a type of biological cell. They have a large domain of prokaryotic microorganisms. Usually a few micrometers long, bacteria have a number of shapes, spreading from spheres to roots and spirals. Bacteria are among the first forms of life that appear on Earth, and are in most habitats.',
 'Bacteria are a type of biological cell. They make up a large area of prokaryotic microorganisms. Usually a few micrometers in length, bacteria have a number of forms, ranging from spheres to rows and spirals. Bacteria were among the first forms of life to appear on Earth, and are present in most of its habitats.',
 'Bacteria are a kind of biological cell. They make a large area of prokaryotic microorganisms. Usually several micrometres of time, bacteria have several varieties, from spheres to color and spirals. Bacteria were among the first kinds of life to appear on the earth, and are present in many parts of its environment.',
 'This is the first time I’ve been able to do it, and I’ve been able to do it, I’ve been able to do it, I’ve been able to do it, I’ve been able to do it, I’ve been able to do it.',
 'Bacteria are a type of biological cells. They generate a large area of prokaryotic microorganisms. Normally, several micrometers, the bacteria have multiple shapes from round to round and round. The bacteria is one of the first forms of life that appears on the earth and is found in most of its environment.',
 'The bacteria are a type of biological cell. These are a large domain of prokaryotic microorganisms. Typically in some micrometres of length, the bacteria have a number of shapes, from spheres to wraps and spirals. The bacteria are one of the first forms of life emerging on Earth, and are available in more habitats.',
 'Go and go and go and go and go and go and go and go.',
 'Bacteria are a biological type of cell. They form a large range of procaryotic microorganisms. typically in a few micrometers length, bacteria have a series of shapes, from spheres to roots and spirals. Bacteria have been among the first forms of life that appear on Earth and are present in most of their living areas.',
 'Bacteria are a type of biological cells. they form a large sphere of procariotic microorganisms. As a rule, a few micrometers of length, bacteria have a number of forms that vary from spheres to cows and spirals. Bacteria were among the first forms of life that appear on Earth, and present in most of its vital objects.',
 'Bacteria are a section of biological cells, they build a large place of procaryotic microorganisms, generally in a range of several micrometers, there are a large number of bacteria shapes, from spiders to wrinkles and spirals, bacteria were one of the first forms of life that appeared on earth, and are present in most populations.',
 'Bacteria are biological cells. They accumulate prokaryotic microorganisms. In a typical micrometer length, bacteria accumulate from sphera to sphera to sphera. Bacteria were one of the first forms of life on Earth, and are present in most biologies.',
 'Bacteria are a type of biological cell. They form a large range of prokaryotic microorganisms. Usually, a few micrometres in length, bacteria have a number of shapes, from crystals to roots and crystals. bacteria are one of the first forms of life appearing on Earth, and present in most of its organisms.',
 'Bacteria is a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria was among the first life forms to appear on Earth, and are present in most of its habitats. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats.',
 'The bacteria is a type of biological cell. The bacteria are prokaryotic microorganisms. According to many micrometres, the bacteria are many forms, from spheres to roads and spirals. The bacteria are the first life forms to appear on Earth, with many habitats.',
 'Bacteria is a type of biological cell. They represent a large area of prokaryotic microorganisms. Types of a little micrometer in the acteria, bacteria have a number of forms, refer to spheres and spirals. Bacteria are among the first life forms to appear on Earth, and are present in most of the bacteria.',
 'Bakteries are a type of biological cell. They were the best of prokaryotic microorganisms. They were one of the many micrometres, the bacteria are a lot of processes, from processes and processes. Bakteries are one of the processes of processes that are processed on Earth, and they are processed in all processes.',
 'Bacteria are a type of biocells, they make up a large area of probiotic microorganisms, usually several meters long bacteria have several shapes, from branches to roots and spirals, bacteria are one of the first forms of life that appear on Earth and are present in most of their habitats.',
 'The bacteria is a type of biological cells. The bacteria is a large area of prokaryotic microorganisms. The bacteria is a long micrometres, the bacteria is a spheres in roads and spirals. The bacteria is among the data, data, data, data and data.']
```

</details>

### Training / Finetuning
Finetune cross-lingual model on your data
```
result = t2t.Fitter().transform(["Hello, World! [TGT] 你好,世界!"],
  src_lang="en",
  tgt_lang="zh",
  num_epochs=10,
  save_directory="model_dir"
)

# load and use model from saved directory
t2t.Transformer.PRETRAINED_TRANSLATOR = "model_dir"
t2t.Translator().transform("Hello, World!", src_lang="en", tgt_lang="zh")
```

### Identification
Identify the language of a text. Not yet accurate for short sequences (<10 tokens)
```
t2t.Identifier().transform(["Aj keď sa Buzz Aldrin stal až „druhým človekom“, ktorý otlačil svoju nohu do povrchu Mesiaca...",])

# Prediction
`['sk', 'Slovak']`
```

### Serving
We aim to serve all functionality above
```
# Start web server
t2t.Server(host='0.0.0.0', port=80)

# Make POST requests
import socket
import requests

address = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
url = f"http://{address}"
transformer = "Translator"
payload = {
  "input_lines": ["hello", "world"],
  "src_lang": "en",
  "tgt_lang": "ko",
}
r = requests.post(f"{url}/{transformer}", json=payload)
print(r.json()) #{'result': ['안녕하세요', '세계']}

# Indexer actions
r = requests.post(f"{url}/Indexer/add", json=payload)
r = requests.post(f"{url}/Indexer/size")
r = requests.post(f"{url}/Indexer/search", json=payload)
r = requests.post(f"{url}/Indexer/retrieve", json=payload)
payload["ids"] = [0,1]
r = requests.post(f"{url}/Indexer/remove", json=payload)
```

## Questions?
For questions or help using Text2Text, please submit a [GitHub issue](https://github.com/artitw/text2text/issues).

## Citation
To cite this work, use the following BibTeX citations.
```
@article{stfidf,
  doi = {10.48550/ARXIV.2209.14281},
  url = {https://arxiv.org/abs/2209.14281},
  author = {Wangperawong, Artit},
  title = {Multilingual Search with Subword TF-IDF},
  publisher = {arXiv},
  year = {2022},
}
```
```
@misc{text2text@2020,
  author={Wangperawong, Artit},
  title={Text2Text: Crosslingual NLP/G toolkit},
  year={2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/artitw/text2text}},
  url = {https://github.com/artitw/text2text}
}
```

## Contributing
There are many ways you can [contribute](https://github.com/artitw/text2text/blob/master/CONTRIBUTING.md):
1. Ask or answer a question in [Issues](https://github.com/artitw/text2text/issues)
2. Share your experiences on using Text2Text
3. Report bugs with information to reproduce
4. Request for new features or functionality
5. Improve code by submitting a [pull request](https://github.com/artitw/text2text/pulls) with outputs demonstrating the change
6. See [projects](https://github.com/artitw/text2text/projects), comment on a task issue to get involved

## Code of Conduct
Please adhere to our [code of conduct](https://github.com/artitw/text2text/blob/master/CODE_OF_CONDUCT.md) when participating in this project.
