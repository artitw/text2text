import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text2text",
    version="0.3.4",
    author="Artit Wangperawong",
    author_email="artitw@gmail.com",
    description="Text2Text: Multilingual tokenization, embedding, search, translation, summarization, question generation, question answering, data augmentation, distance measurement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artitw/text2text",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='multilingual cross-lingual bert natural language processing nlp nlg text generation question answer answering information retrieval tfidf tf-idf search summary summarizer summarization tokenizer tokenization translation backtranslation data augmentation science machine learning colab embedding levenshtein sub-word edit distance',
    install_requires=[
        'numpy',
        'scipy',
        'sentencepiece',
        'torch',
        'tqdm',
        'transformers',
    ],
)