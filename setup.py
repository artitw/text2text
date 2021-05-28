import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text2text",
    version="0.1.6",
    author="Artit Wangperawong",
    author_email="artitw@gmail.com",
    description="Text2Text: Multilingual text transformer for translations, summarizations, questions, answers and variations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artitw/text2text",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='multilingual cross-lingual bert natural language processin nlp nlg text generation question answer summary summarization translation data augmentation science machine learning colab',
    install_requires=[
        'torch',
        'tqdm',
        'numpy',
        'sentencepiece',
        'transformers'
    ],
)