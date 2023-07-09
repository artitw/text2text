import pytest

from text2text.langchain.text2text_assistant import Text2TextAssistant

@pytest.mark.requires("langchain")
def test_llm_inference() -> None:
    input_text = 'Say "hello, world" back to me'
    llm = Text2TextAssistant()
    result = llm(input_text)
    assert "hello" in result.lower() 
