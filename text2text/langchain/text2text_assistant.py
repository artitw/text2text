from typing import Any, List, Mapping, Optional

import text2text as t2t
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class Text2TextAssistant(LLM):
    model: t2t.Assistant = t2t.Assistant()

    @property
    def _llm_type(self) -> str:
        return "Text2Text"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.model.transform([prompt], **kwargs)[0]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"type": self._llm_type}