import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

from ldp_rag import NormalRAG, HybridRAG
from ldp_rag.ldp import LPRAG, Entity
from ldp_rag.privacy_gate import PrivacyGateRAG, privacy_gate_pipeline
from ldp_rag.evaluation import iterate_question_files, run_evaluation
from ldp_rag.chunking import AdaptiveChunkSplitter
from ldp_rag.prompting import redact, typedholder
from ldp_rag.judges import llm_leakage_score
from ldp_rag.metrics import cold_stats_leakage


class Embeddings:
    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return [[1., 0.] for _ in texts]

    def embed_query(self, text):
        return [1., 0.]

    def encode(self, texts, **kwargs):
        return [[0., 0.] for _ in texts]


class LLM:
    def __init__(self, response='answer'):
        self.response, self.messages = response, []

    def invoke(self, messages):
        self.messages.append(messages)
        return Obj(content=self.response)


class Store:
    def __init__(self):
        self.k, self.docs = None, []

    def similarity_search(self, question, k):
        self.k = k
        return [Obj(page_content=f'chunk {i}') for i in range(k)]

    def add_documents(self, documents):
        self.docs.extend(documents)


class Collection:
    def __init__(self):
        self.texts, self.ids = [], []

    def count(self):
        return len(self.texts)

    def add(self, ids, documents, embeddings):
        self.ids.extend(ids)
        self.texts.extend(documents)

    def query(self, query_embeddings, n_results):
        return {'documents': [self.texts[:n_results]]}


class LibraryTests(unittest.TestCase):
    def rag(self, cls=NormalRAG, **kwargs):
        return cls(embeddings=Embeddings(), llm=LLM(), vectorstore=Store(), **kwargs)

    def ldp(self, **kwargs):
        with self.assertWarns(UserWarning):
            return LPRAG(embeddings=Embeddings(), llm=LLM(), nlp=lambda text: Obj(ents=[]),
                         local_embedder=Embeddings(), collection=Collection(), **kwargs)

    def test_normal_and_hybrid_use_actual_top_k(self):
        for cls in (NormalRAG, HybridRAG):
            rag = self.rag(cls)
            answer, context = rag.generate('question', top_k=2, include_context=True)
            self.assertEqual((answer, len(context), rag.vectorstore.k), ('answer', 2, 2))
            self.assertIn('chunk 1', str(rag.llm.messages))

    def test_custom_system_prompt(self):
        rag = self.rag(HybridRAG)
        rag.generate('q', system_prompt='custom')
        self.assertEqual(rag.llm.messages[0][0]['content'], 'custom')

    def test_retrieval_validation(self):
        rag = self.rag()
        for k in (0, -1, 1.2, True):
            with self.assertRaises(ValueError):
                rag.retrieve_context('q', k)
        rag.vectorstore = None
        with self.assertRaises(RuntimeError):
            rag.generate('q')

    def test_faiss_requires_explicit_trust(self):
        with TemporaryDirectory() as directory:
            Path(directory, 'index').mkdir()
            rag = self.rag(vector_store_path=directory)
            with self.assertRaisesRegex(ValueError, 'pickle'):
                rag.load_vector_store('index')
            with self.assertRaises(ValueError):
                rag._index_path('../index')

    def test_gate_prompt_includes_previous_output(self):
        llm = LLM('{"rewritten":"general text"}')
        results = privacy_gate_pipeline('secret', 'SENSITIVE', [1., .1], 2, llm=llm)
        self.assertEqual(len(results[.1]), 3)
        self.assertIn('general text', llm.messages[-1][-1]['content'])

    def test_gate_rewrites_generation_context(self):
        rag = self.rag(PrivacyGateRAG, gate_llm=LLM('{"rewritten":"sanitized"}'))
        answer, context = rag.generate('q', include_context=True)
        self.assertEqual(context, ['sanitized'] * 4)
        self.assertNotIn('chunk 0', str(rag.llm.messages))

    def test_gate_failures_propagate(self):
        llm = LLM('invalid')
        with self.assertRaises(ValueError):
            privacy_gate_pipeline('secret', 'SENSITIVE', llm=llm)
        for schedule in ([], [.1, 1], [1, 1], [math.nan]):
            with self.assertRaises(ValueError):
                privacy_gate_pipeline('secret', 'SENSITIVE', schedule, llm=llm)

    def test_ldp_perturbs_before_remote_embedding_and_indexing(self):
        rag = self.ldp(text_splitter=Obj(split_documents=lambda docs: docs))
        with patch.object(rag, '_perturb_document', return_value='safe'):
            rag.add_documents([Obj(page_content='secret')])
            rag.add_documents([Obj(page_content='secret')])
        self.assertEqual(rag.embeddings.seen, ['safe', 'safe'])
        self.assertEqual(rag.collection.texts, ['safe', 'safe'])
        self.assertEqual(len(set(rag.collection.ids)), 2)
        self.assertEqual(rag.retrieve_context('q', 999), ['safe', 'safe'])

    def test_zero_norm_budget_and_extreme_epsilon(self):
        rag = self.ldp()
        ents = [Entity('a', 'WORD', 0, 1), Entity('b', 'WORD', 2, 3)]
        rag._allocate_budgets(ents)
        self.assertAlmostEqual(sum(e.epsilon for e in ents), rag.total_epsilon)
        self.assertTrue(all(e.epsilon > 0 for e in ents))
        self.assertEqual(rag._rr_word('hello', 10000), 'hello')
        for eps in (1e-8, 1, 10000):
            self.assertTrue(math.isfinite(rag._pm_continuous(.5, eps)))
        with self.assertRaises(ValueError):
            rag._pm_continuous(0, 0)

    def test_entity_spans_are_preserved_and_validated(self):
        rag = self.ldp()
        with patch.object(rag, '_identify_entities', return_value=[Entity('Bob', 'WORD', 3, 6)]), patch.object(rag, '_perturb_entity', return_value='X'):
            self.assertEqual(rag._perturb_document('Hi Bob!'), 'Hi X!')
        with patch.object(rag, '_identify_entities', return_value=[Entity('x', 'WORD', 0, 5), Entity('x', 'WORD', 2, 4)]):
            with self.assertRaises(ValueError):
                rag._perturb_document('abcdef')

    def test_recursive_unicode_evaluation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory, 'questions', 'Healthcare')
            root.mkdir(parents=True)
            (root / '1.json').write_text(json.dumps([{'question': 'سلام', 'reference': 'answer'}]), encoding='utf-8')
            output = Path(directory, 'runs', 'out.jsonl')
            count = run_evaluation(self.rag(), root.parent, output, metrics={'custom': lambda **kw: int(kw['question'] == 'سلام')})
            result = json.loads(output.read_text())
            self.assertEqual(count, 1)
            self.assertEqual(result['question_type'], 'Healthcare')
            self.assertEqual(result['metrics'], {'custom': 1})
            self.assertIn('سلام', output.read_text())

    def test_invalid_dataset_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, 'bad.json')
            path.write_text('{"question":"not a list"}')
            with self.assertRaisesRegex(ValueError, 'list'):
                list(iterate_question_files(directory))
            with self.assertRaises(ValueError):
                run_evaluation(self.rag(), directory, path)

    def test_chunk_boundaries_and_hard_limit(self):
        splitter = AdaptiveChunkSplitter(embedder=Embeddings(), chunk_size=10)
        chunks = splitter.split_text('One. Two. ThisIsAVeryLongSentence.')
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))
        self.assertEqual(''.join(chunks).replace(' ', ''), 'One.Two.ThisIsAVeryLongSentence.')

    def test_redaction_uses_offsets(self):
        nlp = lambda text: Obj(ents=[Obj(text='Ann', start_char=0, end_char=3, label_='PERSON')])
        self.assertEqual(redact('Ann Annex', nlp=nlp), 'IIIIII Annex')
        self.assertEqual(typedholder('Ann Annex', nlp=nlp), 'PERSON Annex')

    def test_leakage_metric(self):
        nlp = lambda text: Obj(ents=[])
        self.assertEqual(cold_stats_leakage('Alice', 'Alice', nlp)['leak_rate'], 1.)
        self.assertEqual(cold_stats_leakage('Alice', 'Nothing', nlp)['leak_rate'], 0.)

    def test_judge_failure_does_not_fabricate_score(self):
        create = lambda **kwargs: Obj(choices=[Obj(message=Obj(content='garbage'))])
        client = Obj(chat=Obj(completions=Obj(create=create)))
        with self.assertRaises(ValueError):
            llm_leakage_score('private', 'public', client=client)


if __name__ == '__main__':
    unittest.main()
