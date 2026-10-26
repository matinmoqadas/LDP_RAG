"""A complete API example requiring no keys or optional dependencies."""
from types import SimpleNamespace
from ldp_rag import NormalRAG


class Retriever:
    def similarity_search(self, query, k=4):
        return [SimpleNamespace(page_content='LDP_RAG is a retrieval research project.')][:k]


class Chat:
    def invoke(self, messages):
        return SimpleNamespace(content='LDP_RAG is a retrieval research project.')


rag = NormalRAG(embeddings=object(), llm=Chat(), vectorstore=Retriever())
print(rag.generate('What is LDP_RAG?', include_context=True))
