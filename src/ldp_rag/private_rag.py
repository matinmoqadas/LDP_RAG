"""RAG ingestion from locally randomized, fixed-vocabulary document reports."""
from dataclasses import dataclass, field
import json

from .accounting import BudgetAccountant
from .mechanisms import TextRandomizer
from .rag import NormalRAG, load_docx


@dataclass
class _ReportDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)
    id: str | None = None


class LocalDPRAG(NormalRAG):
    """Randomize whole documents locally, then split and embed only reports.

    Adjacency replaces text in a fixed public document roster. Report counts,
    queries and caller-selected public configuration are not protected. Each
    document costs randomizer.epsilon; repeated documents from one person must
    use the same privacy_unit and durable ledger. See docs/PRIVACY.md.
    """

    def __init__(self, *args, randomizer: TextRandomizer,
                 accountant: BudgetAccountant, **kwargs):
        if not isinstance(randomizer, TextRandomizer):
            raise TypeError('randomizer must be a TextRandomizer')
        if not isinstance(accountant, BudgetAccountant):
            raise TypeError('accountant must be a BudgetAccountant')
        self.randomizer = randomizer
        self.accountant = accountant
        super().__init__(*args, **kwargs)

    def add_documents(self, documents, *, privacy_unit: str):
        documents = list(documents)
        if not documents:
            raise ValueError('At least one document is required')
        if any(not isinstance(getattr(doc, 'page_content', None), str) for doc in documents):
            raise TypeError('Each document must expose string page_content')
        # Reserve the whole batch before randomization, model calls or writes.
        # A partial failure can already have exposed a report, so never refund.
        self.accountant.reserve(privacy_unit, len(documents) * self.randomizer.epsilon)
        reports = [_ReportDocument(self.randomizer.randomize_text(doc.page_content))
                   for doc in documents]
        # Drop all raw metadata and IDs. Splitting after randomization prevents
        # overlapping raw chunks from silently consuming multiple budgets.
        return super().add_documents(reports)

    def load_documents(self, dataset_folder, *, privacy_unit: str):
        return self.add_documents(load_docx(dataset_folder), privacy_unit=privacy_unit)

    def _policy(self):
        return {'mechanism': 'fixed-token-rr-v1', 'epsilon': str(self.randomizer.epsilon),
                'max_tokens': self.randomizer.max_tokens,
                'vocabulary': list(self.randomizer.vocabulary)}

    def save_vector_store(self, index_name='faiss_index'):
        super().save_vector_store(index_name)
        path = self._index_path(index_name) / 'ldp-policy.json'
        path.write_text(json.dumps(self._policy(), ensure_ascii=False), encoding='utf-8')

    def load_vector_store(self, index_name='faiss_index', *, allow_dangerous_deserialization=False):
        path = self._index_path(index_name) / 'ldp-policy.json'
        if not path.is_file() or json.loads(path.read_text(encoding='utf-8')) != self._policy():
            raise ValueError('Index lacks a matching LocalDPRAG public policy manifest')
        super().load_vector_store(index_name, allow_dangerous_deserialization=allow_dangerous_deserialization)
