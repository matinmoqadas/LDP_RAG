# Code review and refinement

Reviewed upstream commit `9b1b7b8030fb454d1d8caeffc25dc3d5f7f1504b` and the
local 0.1 package. This review concerns the privacy mechanism as well as Python
packaging. Packaging a perturbation algorithm does not establish LDP.

## Findings

1. **Critical — input-dependent word support.** In
   [Models/LDP_RAG.py](https://github.com/matinmoqadas/LDP_RAG/blob/9b1b7b8030fb454d1d8caeffc25dc3d5f7f1504b/Models/LDP_RAG.py),
   `_dummy_neighbours` generates lowercase strings of the input's length.
   All outputs for `a` have length one; all outputs for `bb` have length two.
   Thus the event “length one” has probabilities 1 and 0. No finite epsilon
   can bound their ratio. Even with equal lengths, `Alice` can be returned for
   input `Alice`, but cannot be generated from `Carol` by lowercase neighbours.
   **Refinement:** a single public domain and identical positive output support
   for every valid input, with exact transition-probability inspection.

2. **Critical — untouched text and structure invalidate document privacy.**
   `_perturb_document` copies everything outside NER spans. A document with no
   recognized entities passes through unchanged. Phrase token counts and entity
   boundaries also depend on the private document. Protecting detected entities
   does not protect arbitrary replacement of a document.
   **Refinement:** `TextRandomizer` maps every document to a fixed number of
   slots, maps unknown tokens to a public marker, randomizes padding too, and
   releases no raw offsets, lengths or entity labels. `LocalDPRAG` drops raw
   IDs and metadata and splits only after randomization.

3. **High — no end-to-end composition accounting.** Adaptive norm-based budgets
   are computed from the private entities, so a fixed-parameter composition
   argument cannot simply be applied to that allocation. Normalized embeddings
   also tend to have equal norms, undermining the intended relevance weighting.
   Repeated indexing, overlapping raw chunks, and multiple documents from a
   person had no shared ledger.
   **Refinement:** public fixed per-slot budgets with exact rational division;
   whole-document randomization before chunking; atomic SQLite reservations
   of the full batch cost for a stable privacy unit. Charges survive failures
   and restarts when the same durable ledger is reused.

4. **High — numeric scale and sampler errors.** Upstream clips raw numeric
   values directly to [-1, 1]; for example, 20 and 200 both become 1. Its equal
   probability choice between unequal outer intervals is not uniform over
   their union. At an endpoint it can select a zero-length interval and create
   an input-dependent point mass. The 0.1 port fixed interval weights, but that
   did not resolve document privacy or finite-precision sampling concerns.
   **Refinement:** the recommended `BoundedNumeric` API uses explicit public
   bounds and a discrete public grid, exact randomized-response probabilities,
   and a debiased estimator for the clipped, quantized mean. It deliberately
   does not claim to implement the continuous piecewise mechanism.

5. **High — Privacy Gate epsilon is not a privacy budget.**
   [privacy_gate_module.py](https://github.com/matinmoqadas/LDP_RAG/blob/9b1b7b8030fb454d1d8caeffc25dc3d5f7f1504b/utils/privacy_gate_module.py)
   prompts an LLM to be vague. No output distribution bound follows from that
   instruction, and the rewrite provider receives raw text. Its original deep
   prompt also omitted the previous text.
   **Refinement:** retain it as an explicitly heuristic baseline, require an
   explicit rewriting model, include the previous text, and propagate failures.

6. **High — library lifecycle defects.** Upstream initializes clients/NLP models
   at import time, contains an incomplete print statement in NormalRAG, points
   the CLI at nonexistent module/class names, enables pickle deserialization
   by default, and uses a Chroma client configuration that does not clearly
   establish durable persistence. LLM judges convert errors into neutral scores.
   **Refinement:** the package provides lazy optional integrations, injected
   dependencies, implemented CLI routes, explicit FAISS trust, PersistentClient
   for the experimental Chroma path, and visible evaluation errors. The new
   LocalDPRAG path also checks a matching public-policy manifest before loading.

## What now has a bounded privacy claim

`RandomizedResponse`, `BoundedNumeric`, and `TextRandomizer` have explicit
output-distribution bounds under their fixed public configurations. The proof
and implementation assumptions are in [PRIVACY.md](PRIVACY.md). The legacy
`LPRAG`/`LDPRAG` aliases and Privacy Gate remain experimental baselines. Changing
their labels or raising warnings cannot repair their privacy argument.

`LocalDPRAG` carries the text-report bound through ingestion, embedding and
retrieval as post-processing, provided the local randomizer is inside the
data owner's trust boundary and the document roster/query schedule is public.
User-level accounting requires the same unit ID and ledger for all that user's
documents/releases. This is replacement privacy conditional on the roster;
it does not hide whether a person or document exists.

## Research direction and limits

Whole-text LDP has severe utility costs. For vocabulary size k including markers,
L slots and total budget E, the ideal per-slot truthful probability is
`exp(E/L)/(exp(E/L)+k-1)`. At k=1002, L=128 and E=5 it is about 0.00104.
The exact implemented probability is available through `probability` and may
be slightly smaller. Such output is unlikely to preserve useful prose.

Use this as a defensible reference implementation, then evaluate retrieval and
answer quality on public benchmarks. A vocabulary or numeric bounds learned
from private data must themselves be made private and their cost accounted for.
If the intended threat model protects only a structured attribute or uses metric
DP, define that adjacency explicitly and compare it as a distinct guarantee.
Do not claim that entity perturbation or semantic similarity alone establishes
whole-document LDP. No empirical leakage score proves differential privacy.

Primary references: [Wang et al., USENIX 2017](https://www.usenix.org/conference/usenixsecurity17/technical-sessions/presentation/wang-tianhao),
[Wang et al., ICDE 2019](https://arxiv.org/abs/1907.00782),
and [Dwork and Roth, composition and post-processing](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf).
