"""STF-IDF Retriever.

Based on https://github.com/artitw/text2text"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class STFIDFRetriever(BaseRetriever):
    index: Any
    docs: List[Document]
    k: int = 4

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        **kwargs: Any,
    ) -> STFIDFRetriever:
        try:
            import text2text as t2t
        except ImportError:
            raise ImportError(
                "Could not import text2text, please install with `pip install "
                "text2text`."
            )

        index = t2t.Indexer().transform(texts)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(index=index, docs=docs, **kwargs)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> STFIDFRetriever:
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts, metadatas=metadatas, **kwargs
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        distances, pred_ids = self.index.search([query], k=self.k)
        return [self.docs[i] for i in pred_ids[0] if i >= 0]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        raise NotImplementedError

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        self.docs += documents
        self.index.add(texts)