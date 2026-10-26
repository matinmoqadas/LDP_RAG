"""Sentence similarity chunking extracted from the upstream notebook approach."""
import math
import re


class AdaptiveChunkSplitter:
    """Split on semantic changes or a hard character limit.

    The embedder must expose encode(list[str]); it receives raw document text.
    Unlike the notebook processor, this splitter does not classify sensitivity.
    """
    def __init__(self, *, embedder=None, model_name="all-MiniLM-L6-v2", chunk_size=1000, similarity_threshold=0.5):
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        if not math.isfinite(similarity_threshold) or not -1 <= similarity_threshold <= 1:
            raise ValueError('similarity_threshold must be in [-1, 1]')
        self.embedder = embedder
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

    def split_text(self, text):
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return []
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.model_name)
        vectors = self.embedder.encode(sentences)
        if len(vectors) != len(sentences):
            raise ValueError('Embedder returned the wrong number of vectors')
        chunks, current = [], ''
        previous = None
        for sentence, vector in zip(sentences, vectors):
            boundary = False
            if previous is not None:
                denominator = math.sqrt(sum(float(v)**2 for v in previous) * sum(float(v)**2 for v in vector))
                cosine = sum(float(a)*float(b) for a, b in zip(previous, vector)) / denominator if denominator else 0
                boundary = cosine < self.similarity_threshold
            if current and (boundary or len(current) + len(sentence) + 1 > self.chunk_size):
                chunks.append(current)
                current = ''
            while len(sentence) > self.chunk_size:
                chunks.append(sentence[:self.chunk_size])
                sentence = sentence[self.chunk_size:]
            if sentence:
                current = (current + ' ' + sentence).strip()
            previous = vector
        if current:
            chunks.append(current)
        return chunks

    def split_documents(self, documents):
        from copy import copy
        chunks = []
        for document in documents:
            for text in self.split_text(document.page_content):
                chunk = copy(document)
                chunk.page_content = text
                chunk.metadata = dict(document.metadata)
                chunks.append(chunk)
        return chunks


def get_splitter(**kwargs):
    return AdaptiveChunkSplitter(**kwargs)
