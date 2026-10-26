"""Real DOCX/FAISS roundtrips; optional dependencies, no models or API calls."""
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from ldp_rag import BudgetAccountant, LocalDPRAG, TextRandomizer
from ldp_rag.cli import main

HAS_RAG = all(find_spec(name) is not None for name in
              ('langchain_core', 'langchain_community', 'langchain_text_splitters', 'faiss', 'docx2txt'))


def write_docx(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, 'w') as archive:
        archive.writestr('[Content_Types].xml', '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>''')
        archive.writestr('_rels/.rels', '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>''')
        archive.writestr('word/document.xml', '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>Alice private-medical-secret</w:t></w:r></w:p></w:body></w:document>''')


def embeddings():
    from langchain_core.embeddings import Embeddings

    class LocalEmbeddings(Embeddings):
        def __init__(self):
            self.seen = []

        def embed_documents(self, texts):
            self.seen.extend(texts)
            return [[1.0, float('public' in text)] for text in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    return LocalEmbeddings()


@unittest.skipUnless(HAS_RAG, 'Install the rag extra for DOCX/FAISS integration tests')
class RealRAGIntegrationTests(unittest.TestCase):
    def test_docx_ingestion_faiss_reload_and_generation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_docx(root / 'docs' / 'nested' / '99.DOCX')
            embedder = embeddings()
            with BudgetAccountant(1, path=root / 'budget.sqlite') as accountant:
                kwargs = dict(randomizer=TextRandomizer(('public', 'policy'), 4, 1), accountant=accountant,
                              embeddings=embedder, llm=Obj(invoke=lambda messages: Obj(content='answer')),
                              vector_store_path=root / 'indexes')
                rag = LocalDPRAG(**kwargs)
                self.assertEqual(rag.load_documents(root / 'docs', privacy_unit='user'), 1)
                self.assertEqual(rag.vectorstore.index.ntotal, 1)
                self.assertEqual(accountant.spent('user'), 1)
                self.assertTrue(embedder.seen)
                self.assertNotIn('Alice', str(embedder.seen))
                rag.save_vector_store('private')
                restored = LocalDPRAG(**kwargs)
                restored.load_vector_store('private', allow_dangerous_deserialization=True)
                answer, contexts = restored.generate('Public question', top_k=1, include_context=True)
                self.assertEqual(answer, 'answer')
                self.assertEqual(len(contexts), 1)
                self.assertEqual(contexts[0], embedder.seen[0])
                self.assertEqual(restored.vectorstore.similarity_search('q', k=1)[0].metadata, {})
                self.assertNotIn(b'Alice', (root / 'indexes' / 'private' / 'index.pkl').read_bytes())

    def test_cli_build_then_reuse_index_without_new_spend(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_docx(root / 'docs' / '1.docx')
            (root / 'questions').mkdir()
            (root / 'questions' / '1.json').write_text('[{"question":"Public question"}]')
            (root / 'vocabulary.json').write_text('["public", "policy"]')
            args = ['--rag-type', 'local_dp', '--docs-dir', str(root / 'docs'),
                    '--questions-dir', str(root / 'questions'), '--output-path', str(root / 'results.jsonl'),
                    '--vocabulary', str(root / 'vocabulary.json'), '--max-tokens', '4',
                    '--privacy-ledger', str(root / 'budget.sqlite'), '--epsilon-limit', '1',
                    '--privacy-unit', 'user', '--llm-name', 'none', '--store-dir', str(root / 'indexes')]
            with patch('ldp_rag.rag.make_embeddings', return_value=embeddings()):
                main(args)
                main(args + ['--trust-index'])
            result = json.loads((root / 'results.jsonl').read_text())
            self.assertIn('[LLM stub]', result['answer'])
            self.assertNotIn('Alice', result['answer'])
            with BudgetAccountant(1, path=root / 'budget.sqlite') as accountant:
                self.assertEqual(accountant.spent('user'), 1)
