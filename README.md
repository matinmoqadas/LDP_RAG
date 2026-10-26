# ldp-rag

An installable Local Differential Privacy library built from
[matinmoqadas/LDP_RAG](https://github.com/matinmoqadas/LDP_RAG).
Version 0.2 adds fixed-domain randomized response, bounded numeric reports,
fixed-size text randomization, durable privacy accounting and a LocalDPRAG
ingestion pipeline. Normal RAG, experimental entity perturbation, Privacy Gate,
prompting baselines and evaluation tools are retained for comparison.

Read the [code review](docs/REVIEW.md) for concrete problems in the original
algorithm and the [privacy contract](docs/PRIVACY.md) for guarantees, proofs,
assumptions and utility limits. The legacy `LPRAG`/`LDPRAG` aliases remain
experimental; use `LocalDPRAG` for the new ingestion path.

Python 3.10+ is required. Importing the package does not initialize models,
read API keys, download weights, or connect to services. Core installation has
no third-party dependencies; install the extras needed by your application.

## Install

From this directory:

```bash
python -m pip install .
# DOCX + FAISS + OpenAI-compatible models:
python -m pip install '.[rag,openai]'
# LPRAG with Chroma, spaCy, local allocator and OpenAI retrieval embeddings:
python -m pip install '.[rag,ldp,openai]'
python -m spacy download en_core_web_sm
# Optional Gemini, local retrieval embeddings, scoring, and JSON5 support:
python -m pip install '.[gemini,local,metrics,json5]'
```

Use a virtual environment. Optional dependencies include compiled libraries;
Python 3.10–3.12 generally offer broader wheel availability than newer Python
versions. `local` includes sentence-transformers for adaptive chunking too.

## Python API

The LDP core requires no optional dependencies:

```python
from decimal import Decimal
from ldp_rag import RandomizedResponse, BoundedNumeric, TextRandomizer, BudgetAccountant

rr = RandomizedResponse(("public", "sensitive", "confidential"), epsilon=1)
category_report = rr.randomize("sensitive")
print(rr.probability("sensitive", "public"))  # exact Fraction

numeric = BoundedNumeric(lower=0, upper=100, bins=11, epsilon=1)
number_report = numeric.randomize(42)         # exact public grid point

text = TextRandomizer(("public", "policy", "medical", "service"),
                      max_tokens=4, epsilon=Decimal("0.5"))
with BudgetAccountant(Decimal("2"), path="privacy.sqlite") as ledger:
    ledger.reserve("stable-local-user-id", text.epsilon)
    text_report = text.randomize_text("medical record for Alice")
```

Domains, bounds, vocabulary and budgets must be **public and fixed independently
of private data**. All output tokens, including padding, are randomized.
No raw text or metadata bypasses randomization. A randomized report can still
equal the true input; LDP is a likelihood bound, not guaranteed redaction. Numeric estimation
targets clipped, quantized values; keep exact Fraction reports for estimation.
Category estimates can be negative because the estimator is unbiased and does
not silently clip. Epsilon zero is supported, but uniform reports cannot be
used for estimation. The per-slot/categorical epsilon range is [0, 64].

Primitive randomizers are stateless. Reserve costs yourself for repeated
releases, or use LocalDPRAG, which does this automatically:

```python
from ldp_rag import LocalDPRAG, TextRandomizer, BudgetAccountant

with BudgetAccountant(2, path="privacy.sqlite") as ledger:
    rag = LocalDPRAG(
        randomizer=TextRandomizer(("public", "policy", "medical", "service"), 4, 1),
        accountant=ledger,
        model_name="gpt-4o-mini",  # install rag,openai extras; OPENAI_API_KEY
    )
    rag.load_documents("./data/Docs", privacy_unit="stable-local-user-id")
    answer, context = rag.generate("Public question", include_context=True)
```

Each document costs the configured document epsilon. All documents belonging
to one person must share a stable unit ID and durable ledger, including retries
and later runs. The batch is charged before sending reports; failures do not
refund it. Reopening the same ledger with a different limit is rejected.
In-memory ledgers account only within their own lifetime. Keep ledger files
local and intact; choosing a new file or unit ID resets tracking.

This provides replacement privacy for text in a fixed public document roster.
Document counts, participation, query text, evaluation references and timing
are outside that guarantee. Large vocabularies and long texts require small
per-slot budgets and can produce poor retrieval utility; see the numerical
example in the review. No claim of production privacy certification is made.

For the baseline pipeline without LDP:

```python
from ldp_rag import NormalRAG

rag = NormalRAG(model_name="gpt-4o-mini")  # OPENAI_API_KEY from environment
rag.load_documents("./data/Docs")         # recursively finds arbitrary *.docx
answer, context = rag.generate("What is the policy?", top_k=4, include_context=True)
rag.save_vector_store("policies")
```

The default endpoint comes from the provider SDK. Set `base_url` explicitly
for an OpenAI-compatible proxy. Gemini requires explicit `model_name` and
`embedding_model_name`, and uses `GOOGLE_API_KEY`. Set `embedding_model="local"`
for local retrieval embeddings (install the `local` extra). These choices can
be mixed; use injected `embeddings`/`llm` objects for separate provider keys.

`NormalRAG`, `HybridRAG`, `LPRAG` (alias `LDPRAG`), and `PrivacyGateRAG` share
`load_documents`, `add_documents`, `retrieve_context`, and `generate`.
`generate` returns text, or `(text, list[str])` with `include_context=True`.
`HybridRAG` retains the upstream instruction-based approach and respects
`top_k` during retrieval. It does not implement sparse/dense fusion.

Pass `embeddings`, `llm`, `vectorstore`, and `text_splitter` to use your own
implementations. See [examples/offline.py](examples/offline.py) for a runnable
example without keys or dependencies:

```bash
python examples/offline.py
```

FAISS indexes include pickle metadata. Loading requires an explicit trust
choice, only for an index you created or otherwise trust:

```python
rag.load_vector_store("policies", allow_dangerous_deserialization=True)
```

## Experimental LPRAG

```python
from ldp_rag import LPRAG

rag = LPRAG(total_epsilon=5.0, chroma_dir="./private-index")
rag.load_documents("./data/Docs")
print(rag.generate("What is the policy?"))
# Chroma persists automatically. Reopen using the same directory, collection,
# embedding provider and embedding model; do not ingest the same corpus twice.
```

This ports the upstream named-entity mining, norm-weighted epsilon allocation,
word randomized response, bounded numeric mechanism and phrase perturbation.
Numeric inputs are clipped to [-1, 1], as upstream; outputs are not on the
original unit scale. Non-numeric dates/quantities become hash bucket labels.
Only perturbed chunks are passed to retrieval embedding and storage. Raw
entities go to the allocator, which defaults to a local SentenceTransformer;
keep any injected allocator and NLP implementation within your trust boundary.
The question itself is sent unchanged to the retrieval embedding provider.
`llm_name="none"` returns a clearly marked prompt stub for debugging.

**This research implementation does not establish document-level differential
privacy.** Entity selection, input-dependent word domains and allocation can
leak information, and unrecognized entities remain in plaintext. The privacy
budget applies per chunk invocation, not cumulatively across repeated ingestion
or overlapping chunks. Do not treat `total_epsilon` as a proven corpus guarantee.

## Privacy Gate and prompting baselines

```python
from ldp_rag.privacy_gate import privacy_gate_pipeline

# Supply a trusted LangChain-compatible chat model explicitly.
versions = privacy_gate_pipeline(
    "Text to rewrite", "SENSITIVE", llm=trusted_rewriter,
    eps_schedule=[1.0, 0.5, 0.1], deep_rounds=2,
)
```

`PrivacyGateRAG(gate_llm=trusted_rewriter, ...)` rewrites retrieved context
before answer generation. Its index and rewriting model receive raw text.
Gate epsilon is a wording control, not a formal privacy parameter. Failed or
malformed responses raise errors rather than returning the original context.

`ldp_rag.prompting` exposes `paraphrase`, `zerogen`, `attrprompt`, `sage`,
`redact`, and `typedholder`. LLM functions require `llm=...`; entity functions
require `nlp=...`. The upstream method named `paraphrase` extracts text verbatim.
`redact` and `typedholder` replace entity spans without replacing unrelated
matching substrings.

## Dataset evaluation and CLI

```bash
ldp-rag --rag-type normal \
  --docs-dir ./data/Docs \
  --questions-dir './data/Attack Question' \
  --output-path ./runs/results.jsonl \
  --model-name gpt-4o-mini
```

Also available as `python -m ldp_rag` and `python main.py` after installation.
Use `--help` for all options; the original underscore flag spellings are
accepted for existing runner flags. Choices are `normal`, `hybrid`, `local_dp`, `ldp`, and
`privacy_gate`. The upstream CLI's unimplemented `adaptive` class is replaced
by `--adaptive-chunking`, a sentence-similarity splitter with a hard character
limit. It uses a local model and does not perform the notebook's sensitivity
classification. `--trust-index` opts into loading an existing trusted FAISS
index. For LPRAG, `--store-dir` and `--index-name` select Chroma directory and
collection; data already in that collection is reused.

For the new LDP path, use a public vocabulary and a durable local ledger:

```bash
ldp-rag --rag-type local_dp \
  --docs-dir ./data/Docs --questions-dir ./questions \
  --output-path ./runs/local-dp.jsonl \
  --vocabulary examples/public_vocabulary.json --max-tokens 16 \
  --document-epsilon 1 --epsilon-limit 10 \
  --privacy-unit stable-local-user-id --privacy-ledger privacy.sqlite \
  --model-name gpt-4o-mini
```

All documents in this invocation are charged to the same privacy unit. The
vocabulary is illustrative; select a public vocabulary appropriate to the task
and evaluate utility. Reuse the ledger and unit on later runs. LocalDPRAG index
loading requires a matching public-policy manifest and `--trust-index`.

Privacy Gate requires `--gate-model` (the model receives raw retrieved text).
For Gemini choose explicit chat and embedding model names. For a local-only
LPRAG debugging run choose `--embedding-model local --llm-name none`.

Question files are JSON lists, recursively discovered below the questions
root. Optional JSON5 parsing requires the `json5` extra. Each row needs a
non-empty `question`; optional fields include `reference`, `label`,
`source_sentence`, and `type`. Without `type`, the first subdirectory supplies
the category. Results stream to JSONL with question, question_type, label,
source_sentence, answer, metrics and file fields. The output file is overwritten;
a failed run can leave completed rows for inspection. Raw context is not saved.

```python
from ldp_rag.evaluation import run_evaluation
from ldp_rag.metrics import bleu

run_evaluation(rag, "./questions", "./runs/scored.jsonl", metrics={"bleu": bleu})
```

Custom metric callbacks accept `predicted`, `reference`, `context`, and
`question` keyword arguments and return JSON-serializable values. CLI metrics
are `bleu` and `rouge`, both requiring references. Failures propagate rather
than silently producing fabricated scores. `make_ragas_metrics(llm=...,
embeddings=...)` provides faithfulness and answer_relevancy callbacks with
explicit RAGAS evaluation models. Install the `metrics` extra for these helpers.

`cold_stats_leakage(reference, answer, nlp=...)` retains upstream weighted
entity/token overlap. Omit `nlp` to lazily load spaCy's English model. It is a
heuristic leakage measure, not a privacy guarantee. `ldp_rag.judges` exposes
`llm_leakage_score` and `pairwise_judge` with an explicit OpenAI-compatible
`client`; these send private reference text to that client's endpoint.

## Validation and provenance

```bash
python -m pip install '.[dev]'
python -m unittest discover -s tests -v
python -m build
```

The core suite covers exact transition-probability bounds, joint text reports,
unbiased estimator expectations, sampler mapping, durable/concurrent budget
accounting and RAG data boundaries, plus the existing pipeline regressions.
With the `rag` extra installed, integration tests exercise real DOCX loading,
FAISS indexing/save/reload and CLI index reuse. They use local test embeddings
and chat doubles; no provider calls or model downloads are needed. Paid model
APIs, experimental Chroma persistence and RAGAS remain outside this validation.
See [VALIDATION.md](VALIDATION.md) for the tested environment and results.

See [UPSTREAM.md](UPSTREAM.md) for the pinned source commit and migration notes.
Research notebooks and the standalone plotting script are retained outside the
installed package. The upstream repository supplied no license; this project
preserves that status and does not assign one on the author's behalf.
