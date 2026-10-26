"""Exact probability, composition, accounting, and boundary regression tests."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, Context, ROUND_CEILING, localcontext
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as Obj
import unittest
from unittest.mock import patch

from ldp_rag import (RandomizedResponse, BoundedNumeric, TextRandomizer,
                     BudgetAccountant, BudgetExceededError, LocalDPRAG)
from ldp_rag.cli import parse_args


class MechanismTests(unittest.TestCase):
    def test_transition_matrices_normalize_and_satisfy_epsilon(self):
        for epsilon in (0, Fraction(1, 10), 1, 8, 64, Decimal('1e-100')):
            for k in (2, 3, 7):
                rr = RandomizedResponse(tuple(map(str, range(k))), epsilon)
                matrix = [[rr.probability(x, y) for y in rr.domain] for x in rr.domain]
                self.assertTrue(all(sum(row) == 1 for row in matrix))
                self.assertTrue(all(p > 0 for row in matrix for p in row))
                largest = max(matrix[x][y] / matrix[z][y]
                              for x, z, y in product(range(k), repeat=3))
                self.assertEqual(largest, rr.likelihood_ratio)
                with localcontext(Context(prec=160)):
                    bound = (Decimal(rr.epsilon.numerator) / Decimal(rr.epsilon.denominator)).exp()
                    ratio = Decimal(largest.numerator) / Decimal(largest.denominator)
                    self.assertLessEqual(ratio, bound)

    def test_integer_sampler_exactly_matches_its_probability_matrix(self):
        # Reduce the fixed weight only in this test so every possible draw can
        # be enumerated; the sampler path and interval mapping are unchanged.
        with patch('ldp_rag.mechanisms._WEIGHT', 3):
            rr = RandomizedResponse(('red', 'green', 'blue'), 1)
            for x, value in enumerate(rr.domain):
                counts = dict.fromkeys(rr.domain, 0)
                for draw in range(rr.denominator):
                    counts[rr._decode_draw(x, draw)] += 1
                for report, count in counts.items():
                    self.assertEqual(Fraction(count, rr.denominator), rr.probability(value, report))

    def test_public_randomize_uses_secure_integer_draw(self):
        rr = RandomizedResponse(('yes', 'no'), 1)
        with patch('ldp_rag.mechanisms.secrets.randbelow', return_value=0) as rng:
            self.assertEqual(rr.randomize('yes'), 'yes')
        rng.assert_called_once_with(rr.denominator)

    def test_weights_do_not_depend_on_callers_decimal_context(self):
        expected = RandomizedResponse(('a', 'b'), Fraction(1, 3)).likelihood_ratio
        with localcontext(Context(prec=2, rounding=ROUND_CEILING)):
            actual = RandomizedResponse(('a', 'b'), Fraction(1, 3)).likelihood_ratio
        self.assertEqual(actual, expected)

    def test_zero_epsilon_is_uniform_and_not_estimable(self):
        rr = RandomizedResponse(('a', 'b', 'c'), 0)
        for x, y in product(rr.domain, repeat=2):
            self.assertEqual(rr.probability(x, y), Fraction(1, 3))
        with self.assertRaises(ValueError):
            rr.estimate_frequencies(['a'])

    def test_unbiased_frequency_estimator_uses_implemented_probabilities(self):
        rr = RandomizedResponse(('a', 'b', 'c'), Fraction(1, 2))
        for x, category in product(rr.domain, repeat=2):
            expected = sum(rr.probability(x, y) * rr.estimate_frequencies([y])[category]
                           for y in rr.domain)
            self.assertEqual(expected, int(x == category))

    def test_numeric_reports_and_unbiased_quantized_mean(self):
        mechanism = BoundedNumeric(10, 20, bins=3, epsilon=1)
        for x in (-100, 10, 12, Fraction(25, 2), 16, 20, 100):
            probabilities = [mechanism.probability(x, report) for report in (10, 15, 20)]
            self.assertEqual(sum(probabilities), 1)
            expected = sum(p * mechanism.estimate_mean([report])
                           for report, p in zip((10, 15, 20), probabilities))
            self.assertEqual(expected, mechanism.quantize(x))
        self.assertEqual(mechanism.quantize(12.5), 15)
        with self.assertRaises(ValueError):
            mechanism.estimate_mean([13])

    def test_numeric_ratio_bound_across_clipped_inputs(self):
        mechanism = BoundedNumeric(-5, 15, 5, Fraction(1, 10))
        inputs = (-100, -5, 0, 2, 5, 10, 15, 100)
        for x, z, y in product(inputs, inputs, (-5, 0, 5, 10, 15)):
            ratio = mechanism.probability(x, y) / mechanism.probability(z, y)
            self.assertLessEqual(ratio, mechanism._rr.likelihood_ratio)

    def test_fixed_text_report_joint_probability_bound(self):
        mechanism = TextRandomizer(('a', 'b'), 2, Fraction(1, 3))
        inputs = ('', 'a', 'b', 'a b', 'b a', 'private name', 'a secret', 'b b b b')
        outputs = tuple(product(mechanism._rr.domain, repeat=2))
        for text in inputs:
            self.assertEqual(sum(mechanism.probability(text, report) for report in outputs), 1)
        largest = max(mechanism.probability(x, report) / mechanism.probability(z, report)
                      for x, z, report in product(inputs, inputs, outputs))
        self.assertLessEqual(largest, mechanism._rr.likelihood_ratio ** 2)
        with localcontext(Context(prec=160)):
            self.assertLessEqual(Decimal(largest.numerator) / Decimal(largest.denominator),
                                 (Decimal(1) / 3).exp())
        for text in inputs:
            report = mechanism.randomize(text)
            self.assertEqual(len(report), 2)
            self.assertTrue(set(report) <= set(mechanism._rr.domain))
            self.assertNotIn('secret', report)

    def test_unknown_mapping_truncation_and_padding(self):
        mechanism = TextRandomizer(('a',), 2, 1)
        self.assertEqual(mechanism._encode(''), ('<PAD>', '<PAD>'))
        self.assertEqual(mechanism._encode('秘密 a extra'), ('<UNK>', 'a'))
        self.assertEqual(mechanism._encode('<PAD>'), ('<UNK>', '<PAD>'))
        self.assertEqual(mechanism._rr.epsilon * 2, mechanism.epsilon)

    def test_configuration_is_frozen_and_copied(self):
        domain = ['a', 'b']
        rr = RandomizedResponse(domain, 1)
        domain.append('c')
        self.assertEqual(rr.domain, ('a', 'b'))
        with self.assertRaises(AttributeError):
            rr.epsilon = 100

    def test_reject_invalid_parameters_without_private_values(self):
        for epsilon in (-1, float('inf'), float('nan'), True, '1', 65):
            with self.assertRaises((ValueError, TypeError)):
                RandomizedResponse(('a', 'b'), epsilon)
        for domain in ((), ('a',), ('a', 'a'), (1, 2), 'ab'):
            with self.assertRaises((ValueError, TypeError)):
                RandomizedResponse(domain, 1)
        for vocabulary in ((), ('<PAD>',), ('<UNK>',), ('a b',), (' a',)):
            with self.assertRaises(ValueError):
                TextRandomizer(vocabulary, 2, 1)
        rr = RandomizedResponse(('a', 'b'), 1)
        with self.assertRaises(ValueError) as caught:
            rr.randomize('private-value')
        self.assertNotIn('private-value', str(caught.exception))
        with self.assertRaises(ValueError):
            BoundedNumeric(0, 1, 3, 1).randomize(float('nan'))


class AccountingTests(unittest.TestCase):
    def test_exact_decimal_composition_no_tolerance_overspend(self):
        with BudgetAccountant(Decimal('.3')) as accountant:
            for _ in range(3):
                accountant.reserve('user', Decimal('.1'))
            self.assertEqual(accountant.spent('user'), Fraction(3, 10))
            with self.assertRaises(BudgetExceededError):
                accountant.reserve('user', Decimal('1e-40'))
            self.assertEqual(accountant.spent('another-user'), 0)

    def test_restart_persistence_and_policy_mismatch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, 'privacy.sqlite')
            with BudgetAccountant(2, path=path) as accountant:
                accountant.reserve('user', Fraction(3, 2))
            with BudgetAccountant(2, path=path) as accountant:
                self.assertEqual(accountant.remaining('user'), Fraction(1, 2))
                with self.assertRaises(BudgetExceededError):
                    accountant.reserve('user', 1)
            with self.assertRaises(ValueError):
                BudgetAccountant(3, path=path)

    def test_concurrent_handles_cannot_double_spend(self):
        with TemporaryDirectory() as directory:
            path = Path(directory, 'privacy.sqlite')
            with BudgetAccountant(1, path=path) as a, BudgetAccountant(1, path=path) as b:
                def release(accountant):
                    try:
                        accountant.reserve('same-user', 1)
                        return True
                    except BudgetExceededError:
                        return False
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(release, (a, b)))
                self.assertEqual(sum(results), 1)
                self.assertEqual(a.spent('same-user'), 1)

    def test_failed_reservation_does_not_change_ledger(self):
        with BudgetAccountant(1) as accountant:
            with self.assertRaises(BudgetExceededError):
                accountant.reserve('user', 2)
            self.assertEqual(accountant.spent('user'), 0)
            with self.assertRaises(ValueError):
                accountant.reserve('user', -1)


class RAGBoundaryTests(unittest.TestCase):
    def make_rag(self, accountant, store, splitter=None):
        return LocalDPRAG(randomizer=TextRandomizer(('public', 'policy'), 3, 1),
                          accountant=accountant, embeddings=object(), llm=object(),
                          vectorstore=store, text_splitter=splitter or Obj(split_documents=lambda docs: docs))

    def test_no_raw_text_metadata_or_identifiers_reach_splitter_or_store(self):
        seen, stored = [], []
        def split(docs):
            seen.extend(docs)
            return docs
        with BudgetAccountant(1) as accountant:
            rag = self.make_rag(accountant, Obj(add_documents=stored.extend), Obj(split_documents=split))
            rag.add_documents([Obj(page_content='Alice medical-secret', metadata={'source': '/Alice.docx'}, id='Alice')], privacy_unit='Alice')
            self.assertEqual(accountant.spent('Alice'), 1)
            self.assertEqual(len(stored), 1)
            self.assertEqual(seen[0].metadata, {})
            self.assertIsNone(seen[0].id)
            self.assertNotIn('Alice', str(stored))
            self.assertNotIn('medical-secret', str(stored))

    def test_budget_exhaustion_prevents_any_second_randomization(self):
        with BudgetAccountant(1) as accountant:
            rag = self.make_rag(accountant, Obj(add_documents=lambda docs: None))
            rag.add_documents([Obj(page_content='first')], privacy_unit='user')
            with patch.object(TextRandomizer, 'randomize_text', side_effect=AssertionError('should not run')):
                with self.assertRaises(BudgetExceededError):
                    rag.add_documents([Obj(page_content='second')], privacy_unit='user')

    def test_batch_reservation_happens_before_any_report_and_no_refund_on_failure(self):
        with BudgetAccountant(1) as accountant:
            rag = self.make_rag(accountant, Obj(add_documents=lambda docs: None))
            with self.assertRaises(BudgetExceededError):
                rag.add_documents([Obj(page_content='a'), Obj(page_content='b')], privacy_unit='user')
            self.assertEqual(accountant.spent('user'), 0)
            with patch.object(TextRandomizer, 'randomize_text', side_effect=RuntimeError('failure')):
                with self.assertRaises(RuntimeError):
                    rag.add_documents([Obj(page_content='a')], privacy_unit='user')
            self.assertEqual(accountant.spent('user'), 1)

    def test_baseline_index_cannot_be_loaded_as_local_dp(self):
        with TemporaryDirectory() as directory, BudgetAccountant(1) as accountant:
            rag = self.make_rag(accountant, Obj())
            rag.vector_store_path = directory
            Path(directory, 'index').mkdir()
            with self.assertRaisesRegex(ValueError, 'manifest'):
                rag.load_vector_store('index', allow_dangerous_deserialization=True)

    def test_policy_manifest_roundtrip_and_mismatch(self):
        with TemporaryDirectory() as directory, BudgetAccountant(1) as accountant:
            store = Obj(save_local=lambda path: Path(path).mkdir(parents=True))
            rag = self.make_rag(accountant, store)
            rag.vector_store_path = directory
            rag.save_vector_store('index')
            path = Path(directory, 'index', 'ldp-policy.json')
            self.assertEqual(json.loads(path.read_text()), rag._policy())
            with patch('ldp_rag.rag.NormalRAG.load_vector_store') as load:
                rag.load_vector_store('index', allow_dangerous_deserialization=True)
                load.assert_called_once()
            path.write_text('{}')
            with self.assertRaises(ValueError):
                rag.load_vector_store('index', allow_dangerous_deserialization=True)

    def test_cli_requires_explicit_public_config_and_ledger(self):
        args = parse_args(['--rag-type', 'local_dp', '--docs-dir', 'docs', '--questions-dir', 'questions',
                           '--output-path', 'results.jsonl', '--vocabulary', 'vocab.json',
                           '--epsilon-limit', '0.3', '--privacy-unit', 'user', '--privacy-ledger', 'ledger.sqlite'])
        self.assertEqual(args.epsilon_limit, Decimal('.3'))


if __name__ == '__main__':
    unittest.main()
