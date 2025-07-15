"""
HybridRAG
---------
A retrieval‑augmented generation class that:

• builds / loads a FAISS vector store of docx files
• supports OpenAI‑compatible chat models *or* Gemini‑Pro
• uses LangChain’s RetrievalQA chain
• exposes a generate() method mirroring NormalRAG
"""

import os
from typing import List, Tuple, Optional

# ─── LangChain imports ────────────────────────────────────────────────────────
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, SystemMessage


class HybridRAG:
    """
    Args
    ----
    llm_name   : \"openai\" | \"gemini\"
    api_key    : provider key (OPENAI_API_KEY or GOOGLE_API_KEY)
    model_name : e.g. \"gpt-4o-mini\" or \"gemini-pro\"
    base_url   : OpenAI‑compatible endpoint (ignored by Gemini)
    """

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        llm_name: str = "openai",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.avalai.ir/v1",
        instruction_prompt: str = (
            "You are a helpful assistant. Answer concisely and rely only on the "
            "provided context. If the context is insufficient, say you don't know."
        ),
    ):
        self.llm_name = llm_name.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.base_url = base_url
        self.instruction_prompt = instruction_prompt

        # ── Embeddings + LLM ------------------------------------------------
        if self.llm_name == "openai":
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY missing.")
            self.embeddings = OpenAIEmbeddings(api_key=self.api_key, base_url=self.base_url)
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                base_url=self.base_url,
                temperature=0.0,
            )
        elif self.llm_name == "gemini":
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY missing.")
            self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.api_key)
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=self.model_name,
                temperature=0.0,
            )
        else:
            raise ValueError("llm_name must be 'openai' or 'gemini'")

        # ── Text splitter & vector store -----------------------------------
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore: Optional[FAISS] = None
        self.vector_store_path = "RAG_Database/hybrid_faiss"  # folder on disk

        # ── RetrievalQA chain ----------------------------------------------
        self.retrieval_qa_chain: Optional[RetrievalQA] = None

        # try load existing index
        self.load_vectorstore()

    # ------------------------------------------------------------------ #
    # 1.  Document ingestion
    # ------------------------------------------------------------------ #
    def load_documents(self, docs_root: str) -> None:
        """
        Walk through `docs_root/<topic>/<n>.docx` structure and ingest files.
        """
        if not os.path.exists(docs_root):
            print(f"[HybridRAG] Docs folder not found: {docs_root}")
            return

        docs = []
        for dirpath, _, filenames in os.walk(docs_root):
            for fn in filenames:
                if fn.lower().endswith(".docx"):
                    loader = Docx2txtLoader(os.path.join(dirpath, fn))
                    docs.extend(loader.load())

        if not docs:
            print("[HybridRAG] No .docx files discovered.")
            return

        chunks = self.text_splitter.split_documents(docs)
        print(f"[HybridRAG] ⨁ Ingested {len(docs)} docs → {len(chunks)} chunks")

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)

        self.save_vectorstore()
        self._setup_retrieval_qa_chain()

    # ------------------------------------------------------------------ #
    # 2.  Vector‑store helpers
    # ------------------------------------------------------------------ #
    def load_vectorstore(self) -> None:
        if not os.path.exists(self.vector_store_path):
            print("[HybridRAG] No vector store on disk – run load_documents()")
            return
        try:
            self.vectorstore = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"[HybridRAG] ✓ Vector store loaded from '{self.vector_store_path}'")
            self._setup_retrieval_qa_chain()
        except Exception as e:
            print(f"[HybridRAG] Failed loading vector store: {e}")

    def save_vectorstore(self) -> None:
        if self.vectorstore is None:
            print("[HybridRAG] No vector store to save.")
            return
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vectorstore.save_local(self.vector_store_path)
        print(f"[HybridRAG] 💾 Vector store saved → {self.vector_store_path}")

    # ------------------------------------------------------------------ #
    # 3.  RetrievalQA chain
    # ------------------------------------------------------------------ #
    def _setup_retrieval_qa_chain(self) -> None:
        if self.llm is None or self.vectorstore is None:
            return

        prompt = PromptTemplate(
            template=(
                self.instruction_prompt
                + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
            input_variables=["context", "question"],
        )

        self.retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        print("[HybridRAG] 🔗 RetrievalQA chain ready.")

    # ------------------------------------------------------------------ #
    # 4.  Public generate() method
    # ------------------------------------------------------------------ #
    def generate(
        self,
        question: str,
        top_k: int = 4,
        include_context: bool = False,
        system_prompt: str = (
            "You are a helpful assistant. Answer concisely and rely only on context."
        ),
    ) -> Tuple[str, List[str]] | str:
        """
        Retrieve context + get answer from the underlying RetrievalQA chain.

        Returns
        -------
        str  |  (answer, context_list)
        """
        if self.retrieval_qa_chain is None:
            self._setup_retrieval_qa_chain()
            if self.retrieval_qa_chain is None:
                raise RuntimeError("RetrievalQA chain not initialised.")

        # Chat models allow injecting a system prompt dynamically
        if hasattr(self.llm, "invoke"):
            self.llm.system_message = SystemMessage(content=system_prompt)

        result = self.retrieval_qa_chain.invoke({"query": question})
        answer = result.get("result", "").strip()
        src_docs = result.get("source_documents", [])
        context_chunks = [d.page_content for d in src_docs][:top_k]

        return (answer, context_chunks) if include_context else answer
