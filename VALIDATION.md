# Validation of 0.2.0

Validated locally on Python 3.14.4 (Linux), 2026-10-26.

- 40 tests pass with the `rag` extra installed, including two real DOCX/FAISS
  integration tests. Without optional dependencies, those two tests skip and
  the 38 core tests pass.
- Exact probability matrices normalize and satisfy the requested likelihood
  bound for tested domains and budgets, including zero and tiny epsilon.
- A reduced-weight sampler is exhaustively enumerated; its distribution matches
  the exact probability API. Estimator expectations are checked with Fractions.
- Joint text-report bounds are enumerated for small public vocabularies.
- SQLite spending survives reopening; concurrent independent connections cannot
  double-spend. Failed releases retain reservations; rejected reservations do
  not change the ledger.
- RAG boundary tests verify that raw text, source paths, IDs and metadata do not
  reach splitting/storage; budget exhaustion stops a release before sampling.
- Real DOCX parsing, FAISS indexing, pickle save/load with explicit trust, policy
  manifests, answer generation with a chat double and CLI index reuse pass.
- All Python sources compile, including compatibility wrappers and analysis code.
- Wheel and source distribution build successfully. All 40 tests also pass
  against the installed 0.2.0 wheel from outside the source checkout; the
  installed console entry point reports `ldp-rag 0.2.0`.

Integration versions: langchain-core 1.7.0, langchain-community 0.4.2,
langchain-text-splitters 1.1.3, faiss-cpu 1.15.1, docx2txt 0.9, NumPy 2.5.4.
Tests use local deterministic embeddings and chat doubles; no private data,
paid APIs, model downloads, or external inference calls are needed. Third-party
LangChain deprecation warnings were observed without test failures.

Paid OpenAI/Gemini APIs, model quality, spaCy/SBERT downloads, experimental
Chroma persistence and RAGAS were not exercised. CI defines a Python 3.10–3.13
core matrix and a Python 3.11 RAG integration job; local execution on 3.14.4 does
not establish those matrix results. No independent cryptographic audit or
machine-checked verification has been performed. See docs/PRIVACY.md for the
precise scope and assumptions of the privacy argument.
