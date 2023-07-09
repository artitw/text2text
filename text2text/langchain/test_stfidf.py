import pytest

from text2text.langchain.stfidf import STFIDFRetriever
from langchain.schema import Document


@pytest.mark.requires("langchain")
def test_from_texts() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    stfidf_retriever = STFIDFRetriever.from_texts(texts=input_texts)
    assert len(stfidf_retriever.docs) == 3


@pytest.mark.requires("langchain")
def test_retrieval() -> None:
    input_texts = ["I have a pen.", "Do you have a pen?", "I have a bag."]
    stfidf_retriever = STFIDFRetriever.from_texts(
        texts=input_texts
    )
    assert len(stfidf_retriever.index.retrieve(["pen"], k=2)[0]) == 2


@pytest.mark.requires("langchain")
def test_from_documents() -> None:
    input_docs = [
        Document(page_content="I have a pen."),
        Document(page_content="Do you have a pen?"),
        Document(page_content="I have a bag."),
    ]
    tfidf_retriever = STFIDFRetriever.from_documents(documents=input_docs)
    assert len(tfidf_retriever.docs) == 3
