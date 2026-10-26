"""Common RAG interface with injectable models and vector stores."""
from pathlib import Path
from .providers import invoke_text, make_embeddings, make_llm

DEFAULT_PROMPT = "You are a helpful assistant that answers using only the provided context."


def load_docx(folder):
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Document folder does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".docx")
    if not paths:
        raise ValueError(f"No .docx files found under {root}")
    from langchain_community.document_loaders import Docx2txtLoader
    return [doc for path in paths for doc in Docx2txtLoader(str(path)).load()]


class NormalRAG:
    def __init__(self, llm_name="openai", api_key=None, model_name=None, base_url=None, *,
                 embeddings=None, llm=None, vectorstore=None, text_splitter=None,
                 vector_store_path="RAG_Database", embedding_model=None, embedding_model_name=None):
        self.llm_name = llm_name
        self.embeddings = embeddings if embeddings is not None else make_embeddings(
            embedding_model or llm_name, api_key=api_key, base_url=base_url, model=embedding_model_name)
        self.llm = llm if llm is not None else make_llm(llm_name, api_key=api_key, base_url=base_url, model=model_name)
        self.vectorstore = vectorstore
        self.text_splitter = text_splitter
        self.vector_store_path = str(vector_store_path)

    def _split(self, documents):
        if self.text_splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return self.text_splitter.split_documents(documents)

    def add_documents(self, documents):
        chunks = self._split(list(documents))
        if not chunks:
            raise ValueError("No non-empty document chunks to index")
        if self.vectorstore is None:
            from langchain_community.vectorstores import FAISS
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def load_documents(self, dataset_folder):
        return self.add_documents(load_docx(dataset_folder))

    def retrieve_context(self, question, top_k=4):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if self.vectorstore is None:
            raise RuntimeError("Load documents or a vector store before retrieval")
        return [doc.page_content for doc in self.vectorstore.similarity_search(question, k=top_k)]

    def generate(self, question, top_k=4, system_prompt=DEFAULT_PROMPT, include_context=False):
        context = self.retrieve_context(question, top_k)
        if not context:
            raise ValueError("No context retrieved")
        prompt = "Context:\n" + "\n\n".join(context) + f"\n\nQuestion: {question}\n\nAnswer concisely:"
        answer = invoke_text(self.llm, prompt, system_prompt)
        return (answer, context) if include_context else answer

    def _index_path(self, index_name):
        if not index_name or Path(index_name).name != index_name or index_name in {".", ".."}:
            raise ValueError("index_name must be a single directory name")
        return Path(self.vector_store_path) / index_name

    def save_vector_store(self, index_name="faiss_index"):
        if self.vectorstore is None:
            raise RuntimeError("No vector store to save")
        path = self._index_path(index_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(path))

    def load_vector_store(self, index_name="faiss_index", *, allow_dangerous_deserialization=False):
        path = self._index_path(index_name)
        if not path.is_dir():
            raise FileNotFoundError(path)
        if not allow_dangerous_deserialization:
            raise ValueError("FAISS metadata uses pickle. Pass allow_dangerous_deserialization=True only for an index you trust")
        from langchain_community.vectorstores import FAISS
        self.vectorstore = FAISS.load_local(str(path), self.embeddings, allow_dangerous_deserialization=True)


class HybridRAG(NormalRAG):
    """The upstream instruction-based RAG, using a shared retrieval/generation path."""
    def __init__(self, *args, instruction_prompt=DEFAULT_PROMPT, **kwargs):
        super().__init__(*args, **kwargs)
        self.instruction_prompt = instruction_prompt

    def generate(self, question, top_k=4, include_context=False, system_prompt=None):
        return super().generate(question, top_k, system_prompt or self.instruction_prompt, include_context)

    save_vectorstore = NormalRAG.save_vector_store
    load_vectorstore = NormalRAG.load_vector_store
