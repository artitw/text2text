import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="text2text",
  version="1.3.0",
  author="Artit Wangperawong",
  author_email="artitw@gmail.com",
  description="Text2Text: Crosslingual NLP/G toolkit",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/artitw/text2text",
  packages=setuptools.find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  keywords='multilingual crosslingual gpt chatgpt bert natural language processing nlp nlg text generation gpt question answer answering information retrieval tfidf tf-idf bm25 search index summary summarizer summarization tokenizer tokenization translation backtranslation data augmentation science machine learning colab embedding levenshtein sub-word edit distance conversational dialog chatbot',
  install_requires=[
    'accelerate',
    'auto-gptq',
    'bitsandbytes',
    'peft',
    'faiss-cpu',
    'flask',
    'langchain',
    'googledrivedownloader',
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'sentencepiece',
    'torch',
    'tqdm',
    'transformers',
  ],
)
