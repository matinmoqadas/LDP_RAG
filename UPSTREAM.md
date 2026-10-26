# Source provenance and migration

Based on https://github.com/matinmoqadas/LDP_RAG at commit
`9b1b7b8030fb454d1d8caeffc25dc3d5f7f1504b` (retrieved 2026-10-26).
Original authorship remains with the repository authors. Upstream did not contain
a license; no license or public redistribution permission is inferred here.
Choose a license before publishing a release or uploading to PyPI.

| Upstream | Library |
| --- | --- |
| Models/NormalRag.py | ldp_rag.NormalRAG |
| Models/HybridRag.py | ldp_rag.HybridRAG |
| Models/LDP_RAG.py:LPRAG | ldp_rag.LPRAG (also LDPRAG) |
| Models/Prompted_based_approaches.py | ldp_rag.prompting |
| utils/privacy_gate_module.py | ldp_rag.privacy_gate |
| utils/Metrics.py | ldp_rag.metrics |
| utils/metrics new.py | ldp_rag.metrics and ldp_rag.judges |
| utils/ILS_RW_beta.py | retained analysis script; leakage core in ldp_rag.metrics |
| utils/Semantic_Adaptive_Chunking.ipynb | retained notebook; new splitter in ldp_rag.chunking |
| main.py | ldp-rag CLI and ldp_rag.evaluation |

The upstream `adaptive` RAG class did not exist. Use an implemented RAG type
with `--adaptive-chunking`. PrivacyGateRAG is a new adapter around the upstream
rewriting method. HybridRAG retains instruction-based retrieval; it does not
implement sparse/dense fusion.

Fixes include valid imports and syntax, no import-time provider initialization,
explicit endpoints, recursive DOCX discovery, shared top_k handling, persistent
Chroma, no raw context debug output, explicit FAISS pickle trust, zero-norm
allocation fallback, stable exponential calculations, length-weighted outer
intervals in the numeric sampler, deep-obfuscation input included in prompts,
and judge failures propagated instead of fabricated neutral scores.

Compatibility wrappers require installing the package. Prompt methods require
explicit llm/nlp; judges require client; RAGAS requires llm and embeddings.
The statistical notebooks and legacy plotting script are standalone research
tools excluded from the wheel and may need additional analysis dependencies.

Version 0.2 adds new modules `mechanisms`, `accounting`, and `private_rag`.
Their categorical, numeric-grid, and fixed-token mechanisms are a redesign with
explicit public domains and composition, not a claim that the upstream NER
mechanism has been proved private. The legacy LPRAG/LDPRAG aliases remain for
research comparisons. See docs/REVIEW.md and docs/PRIVACY.md for the review,
proof obligations, deployment assumptions, and the text-utility tradeoff.
