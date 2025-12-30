from typing import Sequence
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class BgeRerank:
    """Класс для повторного ранжирования документов с использованием модели BGE (BAAI General Embedding)"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", top_n: int = 5):
        """
        Инициализация класса

        Args:
            model_name: Название модели для повторного ранжирования
            top_n: Количество возвращаемых наиболее релевантных результатов
        """
        self.model = CrossEncoder(model_name, device="cpu")
        self.top_n = top_n

    def bge_rerank(self, query: str, docs: list[str]) -> list[tuple[int, float]]:
        """
        Выполняет повторное ранжирование документов на основе их релевантности запросу

        Args:
            query: Поисковый запрос, для которого определяется релевантность
            docs: Список текстов документов для ранжирования

        Returns:
            list[tuple[int, float]]: Список кортежей, где каждый кортеж содержит:
                - индекс документа в исходном списке
                - оценка релевантности (чем выше, тем релевантнее)
                Список отсортирован по убыванию релевантности и обрезан до top_n
        """
        model_inputs = [[query, doc] for doc in docs]
        scores = self.model.predict(model_inputs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:self.top_n]

    def compress_documents(self, documents: Sequence[Document], query: str) -> Sequence[Document]:
        if len(documents) == 0:
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        results = self.bge_rerank(query, _docs)
        final_results = []
        for r in results:
            doc = doc_list[r[0]]
            doc.metadata["relevance_score"] = r[1]
            final_results.append(doc)
        return final_results