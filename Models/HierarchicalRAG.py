import os, json, math
from typing import List, Dict, Tuple, Optional
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from sklearn.metrics.pairwise import cosine_similarity   # only used for optional hierarchy tricks

# ---------------------------------------------------------------------
#  EXTENDS YOUR Normal_RAG BUT LEAVES THE PUBLIC API 100 % IDENTICAL
# ---------------------------------------------------------------------
class Hierarchical_RAG:
    """
    • Uses the *same* vector‑store retrieval (vectorstore.similarity_search)
      as your Normal_RAG so nothing downstream breaks.
    • Adds an in‑memory hierarchy so you can experiment with topic/section
      boosts later – but it is totally optional during retrieval.
    """

    # ------- Initialisation boiler‑plate (unchanged) ------------------
    def __init__(self,
                 llm_name: str = "openai",
                 embedding_model: str = "openai",
                 api_key: str = None,
                 model_name: str = None,
                 base_url: str = None):
        self.llm_name = llm_name
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url or "https://api.avalai.ir/v1"

        # -- LLM
        if llm_name == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required.")
            self.llm = ChatOpenAI(api_key=api_key,
                                  model_name=model_name,
                                  base_url=self.base_url)
        elif llm_name == "gemini":
            if not api_key:
                raise ValueError("Gemini API key is required.")
            self.llm = ChatGoogleGenerativeAI(google_api_key=api_key,
                                              model=model_name)
        else:
            raise ValueError("llm_name must be 'openai' or 'gemini'.")

        # -- Embeddings
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings(api_key=api_key,
                                               base_url=self.base_url)
        elif embedding_model == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=api_key)
        else:
            raise ValueError("embedding_model must be 'openai' or 'gemini'.")

        # -- Vector store & helpers
        self.vectorstore: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        self.vector_store_path = "RAG_Database"

        # -- Extra: keep a lightweight topic/section map
        #    {topic: {embedding: [float], sections:{section:{embedding:..}}}}
        self.hierarchy: Dict = {}

    # ------------------------------------------------------------------
    #                       DATA INGESTION                              #
    # ------------------------------------------------------------------
    def json_name(self,rag_model_name):
        json_name = f"{rag_model_name}_{self.llm_name}_{self.embeddings.__class__.__name__}.jsonl"
        return json_name
    def load_documents(self, dataset_folder: str) -> None:
        """
        Same folder convention you already use:
        dataset_folder/
            ├── Topic_A/
            │     ├── 1.docx
            │     ├── 2.docx
            │     └── ...
            └── Topic_B/
                  └── ...
        """
        if not os.path.exists(dataset_folder):
            print(f"Dataset folder '{dataset_folder}' not found.")
            return

        chunks, metadatas = [], []

        for topic in sorted(os.listdir(dataset_folder)):
            topic_path = os.path.join(dataset_folder, topic)
            if not os.path.isdir(topic_path):
                continue

            # -- embed topic once
            self.hierarchy[topic] = {
                "embedding": self.embeddings.embed_query(topic),
                "sections": {}
            }

            # Each sub‑folder is treated as a *section* OR, if the topic
            # folder already contains .docx files directly, we fabricate
            # a single section named "_root".
            subdirs = [d for d in os.listdir(topic_path)
                       if os.path.isdir(os.path.join(topic_path, d))]
            has_direct_docs = any(
                f.lower().endswith(".docx") for f in os.listdir(topic_path))

            if has_direct_docs:
                subdirs.append("_root")   # synthetic section

            for section in sorted(subdirs):
                section_path = (topic_path if section == "_root"
                                else os.path.join(topic_path, section))
                self.hierarchy[topic]["sections"][section] = {
                    "embedding": self.embeddings.embed_query(section)
                                 if section != "_root"
                                 else self.hierarchy[topic]["embedding"],
                }

                # ---- load every *.docx in section_path ----------------
                for fname in os.listdir(section_path):
                    if not fname.lower().endswith(".docx"):
                        continue
                    loader = Docx2txtLoader(os.path.join(section_path, fname))
                    docs = loader.load()
                    for d in self.text_splitter.split_documents(docs):
                        chunks.append(d.page_content)
                        metadatas.append({
                            "topic": topic,
                            "section": section
                        })

        # -- build / extend FAISS index --------------------------------
        if not chunks:
            print("No chunks found; check your folder structure.")
            return

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(chunks,
                                                self.embeddings,
                                                metadatas=metadatas)
        else:
            self.vectorstore.add_texts(chunks, metadatas)

        print(f"Loaded {len(chunks)} chunks from '{dataset_folder}'.")

    # ------------------------------------------------------------------
    #                        RETRIEVAL (unchanged)                      #
    # ------------------------------------------------------------------
    def retrieve_context(self, question: str, top_k: int = 4):
        """
        Identical logic to Normal_RAG – simple vector similarity search.
        Extra hierarchy is *not* used unless you want to modify this later.
        """
        if self.vectorstore is None:
            print("No documents loaded yet.  Call load_documents() first.")
            return []
        docs = self.vectorstore.similarity_search(question, k=top_k)
        return [d.page_content for d in docs]

    # ------------------------------------------------------------------
    #                       LLM wrapper (unchanged)                     #
    # ------------------------------------------------------------------
    def generate(self,
                 question: str,
                 top_k: int = 4,
                 system_prompt: str = (
                     "You are a helpful assistant that answers "
                     "using only the provided context."),
                 include_context: bool = False):
    
        # 1) retrieve
        ctx = self.retrieve_context(question, top_k=top_k)
        if not ctx:
            raise ValueError("No context retrieved.")
    
        # 2) build prompt  ← FIX: pre‑compute context_block
        context_block = "\n\n".join(ctx)
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            f"Answer in a concise and complete manner:"
        )
    
        # 3) call LLM
        msgs = [SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)]
        answer = self.llm.invoke(msgs).content.strip()
        return [answer, ctx] if include_context else [answer]

    # ------------------------------------------------------------------
    #                (optional) save / load hierarchy                   #
    # ------------------------------------------------------------------
    def save_hierarchy(self, outfile: str = "hierarchy.json"):
        with open(outfile, "w", encoding="utf‑8") as f:
            json.dump(self.hierarchy, f)
        print(f"Hierarchy saved → {outfile}")

    def load_hierarchy(self, infile: str = "hierarchy.json"):
        if not os.path.exists(infile):
            print(f"{infile} not found.")
            return
        with open(infile, "r", encoding="utf‑8") as f:
            self.hierarchy = json.load(f)
        print(f"Hierarchy loaded ← {infile}")
