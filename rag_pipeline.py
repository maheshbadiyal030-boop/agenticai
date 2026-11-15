from __future__ import annotations

import hashlib
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class LocalHashEmbeddings(Embeddings):
    """Offline feature-hashing embeddings for a self-contained demonstration."""

    dimensions = 384

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class RAGPipeline:
    def __init__(self, records: list[dict[str, str]], api_key: str = "") -> None:
        if not records:
            raise ValueError("At least one interview record is required to build the RAG index.")
        if api_key:
            from langchain_openai import OpenAIEmbeddings

            embeddings: Embeddings = OpenAIEmbeddings(api_key=api_key)
            self.embedding_mode = "OpenAI embeddings"
        else:
            embeddings = LocalHashEmbeddings()
            self.embedding_mode = "local feature hashing (offline demo)"
        documents = [
            Document(
                page_content=self._record_text(record),
                metadata={
                    "company": record["company"],
                    "job_role": record["job_role"],
                    "category": record["interview_category"],
                    "difficulty": record["difficulty_level"],
                    "source": "historical archive",
                },
            )
            for record in records
        ]
        self.store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="interview_archive",
        )

    @staticmethod
    def _record_text(record: dict[str, str]) -> str:
        return (
            f"Company: {record['company']}\n"
            f"Job role: {record['job_role']}\n"
            f"Category: {record['interview_category']}\n"
            f"Difficulty: {record['difficulty_level']}\n"
            f"Question: {record['interview_question']}\n"
            f"Expected answer points: {record['expected_answer_points']}\n"
            f"Historical candidate feedback: {record['historical_candidate_feedback']}\n"
            f"Common mistakes: {record['common_mistakes']}"
        )

    def retrieve(
        self,
        query: str,
        company: str,
        job_role: str,
        category: str = "Mixed",
        difficulty: str = "Medium",
        k: int = 8,
    ) -> list[Document]:
        filter_terms = [
            {"company": company},
            {"job_role": job_role},
            {"difficulty": difficulty},
        ]
        if category != "Mixed":
            filter_terms.append({"category": category})
        matching = self.store.similarity_search(
            query, k=k, filter={"$and": filter_terms}
        )
        if matching:
            return matching
        filter_terms = [{"company": company}, {"job_role": job_role}]
        if category != "Mixed":
            filter_terms.append({"category": category})
        matching = self.store.similarity_search(
            query, k=k, filter={"$and": filter_terms}
        )
        if matching:
            return matching
        matching = self.store.similarity_search(
            query,
            k=k,
            filter={"$and": [{"company": company}, {"job_role": job_role}]},
        )
        if matching:
            return matching
        return self.store.similarity_search(query, k=k)

    @staticmethod
    def parse_document(document: Document) -> dict[str, str]:
        fields: dict[str, str] = {}
        labels = {
            "Company": "company",
            "Job role": "job_role",
            "Category": "interview_category",
            "Difficulty": "difficulty_level",
            "Question": "interview_question",
            "Expected answer points": "expected_answer_points",
            "Historical candidate feedback": "historical_candidate_feedback",
            "Common mistakes": "common_mistakes",
        }
        for line in document.page_content.splitlines():
            label, separator, value = line.partition(": ")
            if separator and label in labels:
                fields[labels[label]] = value
        fields.update(
            {
                "company": str(document.metadata.get("company", "")),
                "job_role": str(document.metadata.get("job_role", "")),
                "interview_category": str(document.metadata.get("category", "")),
                "difficulty_level": str(document.metadata.get("difficulty", "")),
                "source": str(document.metadata.get("source", "")),
            }
        )
        return fields

    def records(self) -> list[dict[str, str]]:
        data = self.store.get(include=["documents", "metadatas"])
        result = []
        for content, metadata in zip(data["documents"], data["metadatas"]):
            result.append(
                self.parse_document(
                    Document(page_content=content, metadata=metadata or {})
                )
            )
        return result
