import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="text2text",
    version="0.0.9",
    author="Artit Wangperawong",
    author_email="artitw@gmail.com",
    description="Text2Text: generate questions and summaries for your texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artitw/text2text",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='bert nlp nlg text generation question summary summarization data science machine learning',
    install_requires=[
        'torch',
        'tqdm',
        'numpy',
    ],
)
