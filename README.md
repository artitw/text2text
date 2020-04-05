# Text2Text: generate questions and summaries for your texts
Input your text and get questions and summaries in return!

### Citation
To cite this work, use the following BibTeX citation.

```
@misc{text2text@2020,
  author={Wangperawong, Artit},
  title={Text2Text: generate questions and summaries for your texts},
  year={2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/artitw/text2text}},
  url = {https://github.com/artitw/text2text}
}
```

## Requirements
* pytorch
* [pytorch-extension](https://github.com/artitw/apex)
* numpy
* few GBs of memory

## Installation
### A PyTorch Extension (APEX)
```
export CUDA_HOME=/usr/local/cuda-10.1
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" pytorch-extension
```

### Text2Text
```
pip install text2text
```

## Examples
### Colab demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LE_ifTpOGO5QJCKNQYtZe6c_tjbwnulR)

### Obtain some texts
```
notre_dame_str = "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student - run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one - page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut."

bacteria_str = "Bacteria are a type of biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria have a number of shapes, ranging from spheres to rods and spirals. Bacteria were among the first life forms to appear on Earth, and are present in most of its habitats."

bio_str = "Biology is the science that studies life. What exactly is life? This may sound like a silly question with an obvious answer, but it is not easy to define life. For example, a branch of biology called virology studies viruses, which exhibit some of the characteristics of living entities but lack others. It turns out that although viruses can attack living organisms, cause diseases, and even reproduce, they do not meet the criteria that biologists use to define life."
```

### Question Generation
```
from text2text.text_generator import TextGenerator
qg = TextGenerator(output_type="question")

qg.predict([
            bio_str,
            bio_str,
            bio_str,
            bio_str,
            bio_str,
            "I will go to school today to take my math exam.",
            "I will go to school today to take my math exam.",
            "Tomorrow is my cousin's birthday. He will turn 24 years old.",
            notre_dame_str,
            bacteria_str,
            bacteria_str,
            bacteria_str,
            "I will go to school today to take my math exam. [SEP] school",
            "I will go to school today to take my math exam. [SEP] exam",
            "I will go to school today to take my math exam. [SEP] math",
          ])
```
#### Generated Questions
Note that the last three answers were controlled by specifying the `[SEP]` token in the input above.
```
[('What is biology the science that studies?', 'life'),
 ('What is the study of life?', 'studies'),
 ('What would you find the question " life "?', 'sound'),
 ('What can viruses do to living organisms?', 'attack'),
 ('What is the study of life?', 'studies'),
 ('Where will I go to to take my math exam?', 'school'),
 ('Where will I go to to take my math exam?', 'school'),
 ("What will my cousin's birthday?", 'turn'),
 ('What type of oversight does The Observer not have?', 'editorial'),
 ('What shape can bacteria be found in?', 'rods'),
 ('What is the typical length of bacteria?', 'micrometres'),
 ('What is the typical length of bacteria?', 'micrometres'),
 ('Where will I go to to take my math exam?', 'school'),
 ('What will I take after school?', 'exam'),
 ('What exam will I take?', 'math')]
```

### Summary Generation
```
from text2text import TextGenerator
sg = TextGenerator(output_type="summary")
sg.predict([notre_dame_str, bacteria_str, bio_str])

["Notre Dame's students run nine student - run outlets . [X_SEP] Scholastic magazine claims to be the oldest continuous collegiate publication in the United States . [X_SEP] The Observer is an independent publication .",
 'Bacteria were among the first life forms to appear on Earth .',
 'biology is the science that studies life .']
```
#### Generated Summaries
```
["Notre Dame's students run nine student - run outlets . [X_SEP] Scholastic magazine claims to be the oldest continuous collegiate publication in the United States . [X_SEP] The Observer is an independent publication .",
 'Bacteria were among the first life forms to appear on Earth .',
 'biology is the science that studies life .']
```

## Questions?
For questions or help using Text2Text, please submit a GitHub issue.

## Acknowledgements
This package is based on [UniLM](https://github.com/microsoft/unilm)
