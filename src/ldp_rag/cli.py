"""Command-line dataset evaluation."""
import argparse
import json
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from .evaluation import run_evaluation
from .rag import NormalRAG, HybridRAG


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run LDP_RAG dataset evaluation')
    parser.add_argument('--version', action='version', version='ldp-rag 0.2.0')
    parser.add_argument('--rag-type', '--rag_type', choices=['normal', 'hybrid', 'local_dp', 'ldp', 'privacy_gate'], required=True,
                        help='local_dp uses fixed-domain LDP; ldp is the legacy experimental entity method')
    parser.add_argument('--docs-dir', '--docs_dir', required=True)
    parser.add_argument('--questions-dir', '--questions_dir', required=True)
    parser.add_argument('--output-path', '--output_path', required=True)
    parser.add_argument('--index-name', '--index_name', default='faiss_index')
    parser.add_argument('--store-dir', default='RAG_Database')
    parser.add_argument('--top-k', '--top_k', type=int, default=4)
    parser.add_argument('--llm-name', '--llm_name', choices=['openai', 'gemini', 'none'], default='openai')
    parser.add_argument('--embedding-model', choices=['openai', 'gemini', 'local'], default='openai')
    parser.add_argument('--embedding-model-name')
    parser.add_argument('--model-name', '--model_name')
    parser.add_argument('--api-key', '--api_key')
    parser.add_argument('--base-url', '--base_url')
    parser.add_argument('--trust-index', action='store_true', help='Allow loading trusted FAISS pickle metadata')
    parser.add_argument('--total-epsilon', type=float, default=5.0)
    parser.add_argument('--adaptive-chunking', '--adaptive_chunking', action='store_true')
    parser.add_argument('--metrics', nargs='*', choices=['bleu', 'rouge'], default=[])
    parser.add_argument('--gate-model', help='Privacy Gate model; receives raw context')
    parser.add_argument('--gate-epsilon', type=float, default=0.3)
    parser.add_argument('--vocabulary', help='Public JSON list of tokens for local_dp')
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--document-epsilon', type=Decimal, default=Decimal('1'), help='Budget per document for local_dp')
    parser.add_argument('--epsilon-limit', type=Decimal, help='Cumulative budget per privacy unit')
    parser.add_argument('--privacy-unit', help='Stable local identifier for all input documents from the same unit')
    parser.add_argument('--privacy-ledger', help='Persistent local SQLite accounting file; reuse across runs')
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.rag_type == 'privacy_gate' and not args.gate_model:
        parser.error('--gate-model is required for privacy_gate')
    if args.rag_type == 'local_dp':
        if not args.vocabulary or args.epsilon_limit is None or not args.privacy_unit or not args.privacy_ledger:
            parser.error('local_dp requires --vocabulary, --epsilon-limit, --privacy-unit and --privacy-ledger')
        if Path(args.privacy_ledger).resolve() == Path(args.output_path).resolve():
            parser.error('--output-path must not overwrite the privacy ledger')
    return args


def main(argv=None):
    args = parse_args(argv)
    with ExitStack() as stack:
        _run(args, stack)


def _run(args, stack):
    # Validate inputs before constructing provider clients.
    from .evaluation import iterate_question_files
    if next(iterate_question_files(args.questions_dir), None) is None:
        raise ValueError('Question dataset is empty')
    options = dict(llm_name=args.llm_name, api_key=args.api_key, model_name=args.model_name,
                   base_url=args.base_url, embedding_model=args.embedding_model,
                   embedding_model_name=args.embedding_model_name)
    if args.adaptive_chunking:
        from .chunking import AdaptiveChunkSplitter
        options['text_splitter'] = AdaptiveChunkSplitter()
    if args.rag_type == 'ldp':
        from .ldp import LPRAG
        rag = LPRAG(**options, chroma_dir=args.store_dir, collection_name=args.index_name, total_epsilon=args.total_epsilon)
        if rag.collection.count() == 0:
            rag.load_documents(args.docs_dir)
    else:
        cls = {'normal': NormalRAG, 'hybrid': HybridRAG}.get(args.rag_type)
        if args.rag_type == 'local_dp':
            from .private_rag import LocalDPRAG
            from .mechanisms import TextRandomizer
            from .accounting import BudgetAccountant
            vocabulary = json.loads(Path(args.vocabulary).read_text(encoding='utf-8'))
            if not isinstance(vocabulary, list):
                raise ValueError('Vocabulary JSON must be a list of public tokens')
            options['randomizer'] = TextRandomizer(tuple(vocabulary), args.max_tokens, args.document_epsilon)
            options['accountant'] = stack.enter_context(BudgetAccountant(args.epsilon_limit, path=args.privacy_ledger))
            cls = LocalDPRAG
        elif cls is None:
            from .privacy_gate import PrivacyGateRAG
            from .providers import make_llm
            cls = PrivacyGateRAG
            options['gate_llm'] = make_llm(args.llm_name, api_key=args.api_key, base_url=args.base_url, model=args.gate_model)
            options['epsilon'] = args.gate_epsilon
        rag = cls(**options, vector_store_path=args.store_dir)
        if (Path(args.store_dir) / args.index_name).is_dir():
            rag.load_vector_store(args.index_name, allow_dangerous_deserialization=args.trust_index)
        else:
            if args.rag_type == 'local_dp':
                rag.load_documents(args.docs_dir, privacy_unit=args.privacy_unit)
            else:
                rag.load_documents(args.docs_dir)
            rag.save_vector_store(args.index_name)
    from . import metrics
    callbacks = {name: getattr(metrics, name) for name in args.metrics}
    count = run_evaluation(rag, args.questions_dir, args.output_path, metrics=callbacks, top_k=args.top_k)
    print(f'Saved {count} records to {args.output_path}')
