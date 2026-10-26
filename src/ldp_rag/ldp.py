"""Experimental entity perturbation ported from Models/LDP_RAG.py.

Input-dependent word domains, entity selection and budget allocation mean this
research implementation does not establish document-level differential privacy.
"""
from __future__ import annotations
import hashlib
import math
import random
import re
import string
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from typing import List

from .rag import NormalRAG


@dataclass
class Entity:
    text: str
    ent_type: str
    start_char: int
    end_char: int
    epsilon: float = 0.0


class LPRAG(NormalRAG):
    def __init__(self, *, total_epsilon=5.0, c=3, llm_name="openai",
                 embedding_model="openai", retrieval_model_name="all-MiniLM-L6-v2",
                 api_key=None, model_name=None, base_url=None, chroma_dir="chroma_store",
                 collection_name="lprag_corpus", device=None, nlp=None,
                 local_embedder=None, collection=None, **kwargs):
        if not math.isfinite(total_epsilon) or total_epsilon <= 0:
            raise ValueError("total_epsilon must be finite and positive")
        if not isinstance(c, int) or isinstance(c, bool) or c < 1:
            raise ValueError("c must be a positive integer")
        warnings.warn("Experimental perturbation; document-level differential privacy is not established", UserWarning, stacklevel=2)
        self.total_epsilon, self.c = total_epsilon, c
        super().__init__(llm_name, api_key, model_name, base_url,
                         embedding_model=embedding_model, **kwargs)
        if nlp is None:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except OSError as exc:
                raise RuntimeError("Install the NLP model: python -m spacy download en_core_web_sm") from exc
        self.nlp = nlp
        if local_embedder is None:
            from sentence_transformers import SentenceTransformer
            local_embedder = SentenceTransformer(retrieval_model_name, device=device)
        self.local_embedder = local_embedder
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.chroma_client = None
        if collection is None:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            collection = self.chroma_client.get_or_create_collection(collection_name)
        self.collection = collection

    def add_documents(self, documents):
        chunks = self._split(list(documents))
        if not chunks:
            raise ValueError("No document chunks to index")
        texts = [self._perturb_document(c.page_content) for c in chunks]
        # Only perturbed text reaches the retrieval embedding provider and store.
        vectors = self.embeddings.embed_documents(texts)
        self.collection.add(ids=[uuid4().hex for _ in texts], documents=texts, embeddings=vectors)
        return len(texts)

    def retrieve_context(self, question, top_k=4):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(query_embeddings=[self.embeddings.embed_query(question)], n_results=min(top_k, count))
        docs = result.get("documents")
        return docs[0] if docs else []

    def save_vector_store(self, index_name=None):
        if index_name is not None:
            raise ValueError("LPRAG persists automatically; configure chroma_dir and collection_name")
        if self.chroma_client is None:
            raise RuntimeError("An injected collection manages its own persistence")

    def load_vector_store(self, index_name=None):
        self.save_vector_store(index_name)
        self.collection = self.chroma_client.get_collection(self.collection_name)

    def json_name(self, rag_model_name):
        return f"{rag_model_name}_{self.llm_name}_{self.embeddings.__class__.__name__}.jsonl"

    _NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

    def _perturb_document(self, text: str) -> str:
        ents = sorted(self._identify_entities(text), key=lambda ent: ent.start_char)
        previous_end = 0
        for ent in ents:
            if ent.start_char < previous_end or not (0 <= ent.start_char < ent.end_char <= len(text)):
                raise ValueError("Entities must have valid, non-overlapping spans")
            previous_end = ent.end_char
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
            vecs = [self.local_embedder.embed_query(e.text) for e in ents]
        if len(vecs) != len(ents):
            raise ValueError("Allocator returned the wrong number of embeddings")
        norms = [math.sqrt(sum(float(v) ** 2 for v in vec)) for vec in vecs]
        if any(not math.isfinite(norm) for norm in norms):
            raise ValueError("Allocator embeddings must be finite")
        total = sum(norms)
        # A zero vector must not create epsilon=0 in downstream mechanisms.
        weights = [n / total for n in norms] if total and all(norms) else [1 / len(ents)] * len(ents)
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
        p_self = 1 / (1 + self.c * math.exp(-eps))
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
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("epsilon must be finite and positive")
        t = math.exp(-eps / 2)
        denominator = -math.expm1(-eps / 2)
        if denominator == 0:
            raise ValueError("epsilon is too small for floating-point perturbation")
        Q = (1 + t) / denominator
        l = (Q + 1) / 2 * x - (Q - 1) / 2
        r = l + Q - 1
        if random.random() < 1 / (1 + t):
            return random.uniform(l, r)
        # Select the outer intervals in proportion to their lengths.
        left, right = l + Q, Q - r
        return random.uniform(-Q, l) if random.random() * (left + right) < left else random.uniform(r, Q)

    def _lh_discrete(self, s: str, eps: float, g: int = 64) -> str:
        bucket_orig = self._hash_int(s) % g
        p = 1 / (1 + (g - 1) * math.exp(-eps))
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




LDPRAG = LPRAG
