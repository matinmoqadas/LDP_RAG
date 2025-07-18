
from __future__ import annotations

import hashlib, math, os, random, re, string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

# ── Chroma -------------------------------------------------------------------
import chromadb
from chromadb.config import Settings

# ── LangChain helpers --------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    OpenAIEmbeddings = ChatOpenAI = None  # type: ignore

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError:
    GoogleGenerativeAIEmbeddings = ChatGoogleGenerativeAI = None  # type: ignore

try:
    from langchain.schema import SystemMessage, HumanMessage
except ImportError:
    class _Msg:                       # type: ignore
        def __init__(self, content: str): self.content = content
    SystemMessage = HumanMessage = _Msg  # type: ignore

# --------------------------------------------------------------------------- #
# Dataclass for entity metadata
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    text: str
    ent_type: str          # WORD | NUMBER | PHRASE
    start_char: int
    end_char: int
    epsilon: float = 0.0

# --------------------------------------------------------------------------- #
# LPRAG with Chroma backend
# --------------------------------------------------------------------------- #
class LPRAG:
    def __init__(
        self,
        *,
        total_epsilon: float = 5.0,
        c: int = 3,
        llm_name: str = "openai",            # "openai" | "gemini" | "none"
        embedding_model: str = "openai",     # "openai" | "gemini" | "local"
        retrieval_model_name: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.avalai.ir/v1",
        chroma_dir: str = "chroma_store",
        collection_name: str = "lprag_corpus",
        device: Optional[str] = None,
    ) -> None:
        # privacy params
        self.total_epsilon = total_epsilon
        self.c = c

        # NLP
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

        # embeddings -------------------------------------------------------
        self.embedding_model = embedding_model.lower()
        self.llm_name = llm_name.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if self.embedding_model == "openai":
            if OpenAIEmbeddings is None:
                raise ImportError("Install langchain-openai or choose embedding_model='local'")
            self.embeddings = OpenAIEmbeddings(api_key=self.api_key, base_url=base_url)
        elif self.embedding_model == "gemini":
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError("Install langchain-google-genai or choose embedding_model='local'")
            self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.api_key)
        else:  # local MiniLM
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=retrieval_model_name, model_kwargs={"device": device or "cpu"})

        # allocator embedder (local SBERT if available)
        if self.embedding_model == "local":
            self.local_embedder = self.embeddings.client  # SentenceTransformer inside wrapper
        else:
            self.local_embedder = SentenceTransformer(retrieval_model_name, device=device)

        # LLM --------------------------------------------------------------
        if self.llm_name == "openai":
            if ChatOpenAI is None:
                raise ImportError("langchain-openai not installed")
            self.llm = ChatOpenAI(api_key=self.api_key, model_name=model_name, base_url=base_url)
        elif self.llm_name == "gemini":
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain-google-genai not installed")
            self.llm = ChatGoogleGenerativeAI(google_api_key=self.api_key, model=model_name)
        else:
            self.llm = None  # stub; generate() will echo prompt

        # text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Chroma client ----------------------------------------------------
        self.chroma_dir = Path(chroma_dir)
        self.chroma_client = chromadb.Client(Settings(persist_directory=str(self.chroma_dir)))
        self.collection = self.chroma_client.get_or_create_collection(collection_name)

        # raw doc vault (never exposed)
        self._private_corpus: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Helper: JSONL filename hint
    # ------------------------------------------------------------------ #
    def json_name(self, rag_model_name: str) -> str:
        return f"{rag_model_name}_{self.llm_name}_{self.embeddings.__class__.__name__}.jsonl"

    # ------------------------------------------------------------------ #
    # 1.  Document ingestion  (patched to perturb before indexing)
    # ------------------------------------------------------------------ #
    def load_documents(self, dataset_folder: str) -> None:
        if not os.path.exists(dataset_folder):
            print(f"Folder not found: {dataset_folder}")
            return

        raw_docs = []
        for category in os.listdir(dataset_folder):
            cat_path = os.path.join(dataset_folder, category)
            if not os.path.isdir(cat_path):
                continue
            for i in range(1, 11):
                fp = os.path.join(cat_path, f"{i}.docx")
                if os.path.exists(fp):
                    raw_docs.extend(Docx2txtLoader(fp).load())

        if not raw_docs:
            print(f"No .docx found under {dataset_folder}")
            return

        # 1 · split, 2 · **perturb**, 3 · embed & add
        chunks = self.text_splitter.split_documents(raw_docs)
        perturbed = [self._perturb_document(c.page_content) for c in chunks]
        embs      = self.embeddings.embed_documents(perturbed)
        ids       = [f"doc_{self.collection.count()+i}" for i in range(len(perturbed))]

        self.collection.add(ids=ids, documents=perturbed, embeddings=embs)
        print(f"Loaded {len(raw_docs)} docs → {len(perturbed)} perturbed chunks.")


    # ------------------------------------------------------------------ #
    # 2.  Retrieval
    # ------------------------------------------------------------------ #
    def retrieve_context(self, query: str, top_k: int = 4) -> List[str]:
        q_emb = self.embeddings.embed_query(query)
        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        return res["documents"][0] if res["documents"] else []

    # ------------------------------------------------------------------ #
    # 3.  Generate answer
    # ------------------------------------------------------------------ #
    def generate(
        self,
        question: str,
        top_k: int = 4,
        *,
        include_context: bool = False,
        system_prompt: str = "You are a helpful assistant that answers using only the provided context.",
    ):
        ctx = self.retrieve_context(question, top_k)
        print(ctx)
        ctx_block = "\n".join(ctx)
        prompt = f"Context:\n{ctx_block}\n\nQuestion: {question}\n\nAnswer concisely:"
        if self.llm is None:
            answer = f"[LLM stub]\n{prompt}"
        else:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
            answer = self.llm.invoke(messages).content.strip()
        return (answer, ctx) if include_context else answer
        
    
    # ------------------------------------------------------------------
    # Differential‑privacy pipeline (entity mining → perturbation)
    # ------------------------------------------------------------------
    _NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

    def _perturb_document(self, text: str) -> str:
        ents = self._identify_entities(text)
        self._allocate_budgets(ents)
        pieces, cursor = [], 0
        for ent in ents:
            pieces.append(text[cursor : ent.start_char])
            pieces.append(self._perturb_entity(ent))
            cursor = ent.end_char
        pieces.append(text[cursor:])
        return "".join(pieces)

    # 1) Entity identification
    def _identify_entities(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        return [
            Entity(e.text, self._label_to_type(e.label_, e.text), e.start_char, e.end_char)
            for e in doc.ents
        ]

    @staticmethod
    def _label_to_type(self_label: str, txt: str) -> str:
        if self_label in {
            "DATE",
            "TIME",
            "PERCENT",
            "MONEY",
            "QUANTITY",
            "CARDINAL",
            "ORDINAL",
        }:
            return "NUMBER"
        if re.fullmatch(LPRAG._NUM_RE, txt):
            return "NUMBER"
        return "PHRASE" if " " in txt else "WORD"

    # 2) Adaptive epsilon allocation (Eq. 3‑4 in the paper)
    def _allocate_budgets(self, ents: List[Entity]) -> None:
        if not ents:
            return
        # Embed each entity – supports both local SBERT and remote embed_query()
        if hasattr(self.local_embedder, "encode"):
            vecs = self.local_embedder.encode([e.text for e in ents], convert_to_numpy=True)
        else:
            vecs = np.vstack([self.local_embedder.embed_query(e.text) for e in ents])
        norms = np.linalg.norm(vecs, axis=1)
        weights = norms / norms.sum()
        for ent, w in zip(ents, weights):
            ent.epsilon = float(w) * self.total_epsilon

    # 3) Entity‑specific perturbation -----------------------------------
    def _perturb_entity(self, ent: Entity) -> str:
        if ent.ent_type == "WORD":
            return self._rr_word(ent.text, ent.epsilon)
        if ent.ent_type == "NUMBER":
            return self._perturb_number(ent.text, ent.epsilon)
        if ent.ent_type == "PHRASE":
            return self._perturb_phrase(ent.text, ent.epsilon)
        return ent.text

    # 3a) Random‑Response for words ------------------------------------
    def _rr_word(self, word: str, eps: float) -> str:
        neighbours = self._dummy_neighbours(word, self.c)
        pool = [word] + neighbours
        p_self = math.exp(eps) / (math.exp(eps) + self.c)
        if random.random() < p_self:
            return word
        return random.choice(pool[1:])

    def _dummy_neighbours(self, word: str, k: int) -> List[str]:
        letters = string.ascii_lowercase
        return ["".join(random.choices(letters, k=len(word))) for _ in range(k)]

    # 3b) Number perturbation ------------------------------------------
    def _perturb_number(self, raw: str, eps: float) -> str:
        try:
            val = float(raw)
            return f"{self._pm_continuous(val, eps):.4f}"
        except ValueError:
            return self._lh_discrete(raw, eps)

    @staticmethod
    def _pm_continuous(x: float, eps: float) -> float:
        # Piece‑wise mechanism for values scaled to [-1,1]
        x = max(-1.0, min(1.0, x))
        Q = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
        l = (Q + 1) / 2 * x - (Q - 1) / 2
        r = l + Q - 1
        if random.random() < math.exp(eps / 2) / (math.exp(eps / 2) + 1):
            return random.uniform(l, r)
        return random.uniform(-Q, l) if random.random() < 0.5 else random.uniform(r, Q)

    def _lh_discrete(self, s: str, eps: float, g: int = 64) -> str:
        bucket_orig = self._hash_int(s) % g
        p = math.exp(eps) / (math.exp(eps) + g - 1)
        bucket = bucket_orig if random.random() < p else random.choice([i for i in range(g) if i != bucket_orig])
        return f"BUCKET_{bucket:02d}"

    @staticmethod
    def _hash_int(s: str) -> int:
        return int(hashlib.sha256(s.encode()).hexdigest(), 16)

    # 3c) Phrase perturbation ------------------------------------------
    def _perturb_phrase(self, text: str, eps: float) -> str:
        tokens = text.split()
        sub_eps = eps / max(1, len(tokens))
        out_tokens = []
        for tok in tokens:
            if re.fullmatch(self._NUM_RE, tok):
                out_tokens.append(self._perturb_number(tok, sub_eps))
            else:
                out_tokens.append(self._rr_word(tok, sub_eps))
        return " ".join(out_tokens)

    # ------------------------------------------------------------------
    # Embedding helper (local SBERT vs remote)
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> List[float]:
        if hasattr(self.embeddings, "embed_query"):
            return self.embeddings.embed_query(text)
        return self.embeddings.encode(text, convert_to_numpy=True).tolist()

