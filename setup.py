import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text2text",
    version="0.1.8",
    author="Artit Wangperawong",
    author_email="artitw@gmail.com",
    description="Text2Text: Multilingual tokenization, translation, summarization, question generation, question answering, text variation, and edit distance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artitw/text2text",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='multilingual cross-lingual bert natural language processing nlp nlg text generation question answer answering summary summarizer summarization tokenizer tokenization translation data augmentation science machine learning colab levenshtein sub-word edit distance',
    install_requires=[
        'torch',
        'tqdm',
        'numpy',
        'sentencepiece',
        'transformers'
    ],
)