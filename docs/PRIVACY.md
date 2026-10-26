# Privacy contract and proof

## Threat model

The local randomizer runs on a trusted client before the collector, embedding
service or vector store sees a report. An attacker observes the returned report
and any downstream processing of it. For all valid inputs x,x' and events S,
the intended bound is `P[M(x) in S] <= exp(epsilon) P[M(x') in S]` (delta=0).
The public configuration, valid input types, participation, number of reports,
queries and release schedule are fixed or public. Timing, exceptions, resource
usage, a compromised client/OS, memory dumps and ledger disclosure are outside
this output-distribution guarantee. Invalid input is rejected locally and must
not be forwarded as a data-dependent remote error channel.

The public categorical domain, numeric bounds/grid, token vocabulary, token
limit and budgets must be fixed independently of the protected data. Private
configuration learning needs its own mechanism and accounting. The package
cannot determine whether a caller's purportedly public vocabulary is public.

## Categorical randomized response

For a domain with k >= 2 values, let b=2^53 and choose integer a >= b with
`a/b <= exp(epsilon)`. For `Z=a+(k-1)b`, the implementation returns x with
probability a/Z and each other category with probability b/Z. Every input has
the same support. For any output y and any x,x', the ratio is at most a/b,
which is at most exp(epsilon). Summing over y proves the event bound.

A report can equal the real input. LDP bounds an observer's evidence; it does
not guarantee that every name or true attribute disappears from every report.

`secrets.randbelow(Z)` selects an integer without modulo bias. The first a
integers map to the input; the remaining k-1 contiguous blocks, each of size b,
map to the other domain entries. No seed or pseudorandom test generator can be
passed through the public API. This argument models the OS-backed generator
as independent uniform randomness; it is not a certification of the OS or its
entropy source.

The implementation preserves epsilon as an exact Fraction. It uses an isolated
80-digit Decimal context, rounds the epsilon conversion downward, takes a
correctly rounded exponential and then its predecessor, and rounds the scaled
weight downward. Hence the computed ratio never rounds above exp(epsilon).
The lower clamp a=b is valid because epsilon>=0. Decimal's correctly rounded
exponential is documented in the [Python library reference](https://docs.python.org/3/library/decimal.html#decimal.Decimal.exp).
The per-mechanism epsilon range is [0,64]. Tiny positive epsilon can yield a=b;
the mechanism then returns uniform, uninformative reports. This is exposed
through `likelihood_ratio`, rather than silently dividing by zero in estimation.

`probability` returns exact Fractions for valid input/output domain values.
`estimate_frequencies` inverts the implemented p=a/Z and q=b/Z using
`(observed_frequency-q)/(p-q)`. It is unbiased for category proportions with
fresh reports from this configuration. It can return negative estimates or
values above one; clipping would introduce bias. At p=q no estimator is offered.

## Numeric values

`BoundedNumeric` deterministically clips finite inputs to public [lower,upper],
then maps them to the nearest of `bins` equally spaced public grid points (ties
upward). It randomizes the index with the same categorical mechanism and
decodes the index to an exact Fraction. For any two raw finite inputs, their
indices are categorical inputs, so the same epsilon bound holds. Clipping and
quantization do not consume extra budget. The estimate targets the clipped,
quantized mean. Quantization error is at most half a grid interval relative to
the clipped mean; there is no such bound relative to unbounded outliers.

Returning exact Fractions keeps output support fixed. Converting a report to
float or decimal text is optional post-processing. For estimation, keep exact
Fractions (or reconstruct with numerator/denominator); rounded float reports
need not coincide exactly with the public grid and are rejected.

## Arbitrary text and composition

`TextRandomizer` deterministically encodes any string into L=max_tokens slots.
Whitespace tokenization is case-sensitive; unknown words map to <UNK>; input
is truncated to L tokens and padded with <PAD>. The public output domain is
the vocabulary plus those two markers. All L slots are randomized independently
with the exact rational budget epsilon/L, including padding and unknown tokens.

For any two strings and any L-slot report, each slot's likelihood ratio is at
most exp(epsilon/L). The joint ratio is bounded by their product, exp(epsilon).
All outputs have the same number of slots. Original lengths, labels, offsets,
unknown words, raw IDs and metadata are not part of the report. The mechanism
does not claim to preserve meaning or detect sensitive entities.

These arguments follow the standard [sequential composition and post-processing
properties](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf), applied to
the local channel. A frozen vocabulary and slot count are essential to them.

## Accounting and RAG

`BudgetAccountant` records exact sums per privacy unit, with no floating-point
tolerance for overspend. A SQLite immediate transaction serializes reservations
across threads and connections. A release's cost is committed before sampling
or sending reports. Failed work does not refund the reservation, because a
remote system may have already received a partial result. Reusing the same file
preserves spending and its configured limit on restart. Reopening with a
different limit is rejected. The ledger has no reset/refund API.

Primitive `randomize` calls are stateless: reserve their cost yourself for
repeated release. `LocalDPRAG.add_documents(..., privacy_unit=...)` reserves
`number_of_documents * document_epsilon` for you. Use the same stable unit ID
for all releases belonging to a person. Grouping each document as a different
unit provides only document-level accounting, not user-level protection. An
in-memory ledger, deleted/rolled-back ledger, changed unit IDs, different
ledger files, or untracked releases defeats cross-run composition accounting.
Caller identity mapping and protecting ledger integrity are deployment duties.

LocalDPRAG sanitizes each whole document, drops all raw metadata and IDs, and
only then invokes splitters, embedders and storage. Overlap in already-randomized
chunks is post-processing; it does not require new randomization or budget.
Repeated answers using only that fixed sanitized index, public queries and
independent model randomness are also post-processing. Queries, reference text
sent to evaluation models, and information already known by an LLM are outside
the corpus privacy guarantee. Raw data must not reach a remote DOCX parser,
custom hook, embedder or prior vector store through other application paths.

Persisted FAISS stores require both explicit pickle trust and a matching public
policy manifest. The manifest prevents accidental configuration mismatch; it
is not cryptographic attestation or proof that someone else's index was
sanitized. A supplied vector store or loaded index must already contain only
authorized public or sanitized data. Do not mix a raw baseline index into this
pipeline. Sanitization does not validate the safety of deserializing pickle.

## Evidence and limitations

Tests exhaust finite probability matrices, enumerate a reduced-weight sampler,
check exact estimator expectations and small joint text spaces, and exercise
durable/concurrent spending and the RAG data boundary. These regressions support
the algebra above; passing tests alone does not prove every implementation or
deployment satisfies the threat model. This library has not received an
independent cryptographic audit or formal machine-checked verification.

Legacy `LPRAG`/`LDPRAG`, its continuous numeric helper, NER-based redaction and
Privacy Gate are research baselines with no whole-document LDP claim. See
[REVIEW.md](REVIEW.md) for counterexamples and the remaining utility tradeoffs.
