"""Finite-output local randomizers with auditable transition probabilities.

Privacy is over the returned report under a fixed, public configuration. The
sampler uses OS-backed randomness and integer weights, not floating uniforms.
See docs/PRIVACY.md for the probability bound and implementation assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_FLOOR, localcontext
from fractions import Fraction
import secrets
from typing import Iterable

Number = int | float | Decimal | Fraction
_WEIGHT = 2**53


def as_fraction(value: Number, name: str) -> Fraction:
    """Preserve the exact supplied number; reject non-finite or boolean values."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, Fraction)):
        raise TypeError(f"{name} must be a finite number")
    try:
        return Fraction(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc


def privacy_budget(value: Number) -> Fraction:
    epsilon = as_fraction(value, 'epsilon')
    if epsilon < 0:
        raise ValueError('epsilon must be non-negative')
    return epsilon


def _self_weight(epsilon: Fraction) -> int:
    # Bound work and representable parameter sizes. Zero is a valid budget.
    if epsilon > 64:
        raise ValueError('Per-mechanism epsilon must be at most 64')
    # A fresh Context makes the guarantee independent of caller Decimal settings.
    with localcontext(Context(prec=80, rounding=ROUND_FLOOR)) as ctx:
        lower_epsilon = Decimal(epsilon.numerator) / Decimal(epsilon.denominator)
        # Decimal.exp is correctly rounded (half-even). One predecessor is a
        # lower bound on the exact exponential; the following product rounds down.
        lower_exp = lower_epsilon.exp().next_minus(ctx)
        return max(_WEIGHT, int(lower_exp * _WEIGHT))


@dataclass(frozen=True)
class RandomizedResponse:
    """Generalized randomized response over a fixed public string domain.

    For k categories, the implemented probabilities are a/Z for the input and
    b/Z otherwise, Z=a+(k-1)b, with 1 <= a/b <= exp(epsilon). Tiny positive
    budgets can round to a=b (uniform, zero information). No seed is accepted.
    """

    domain: tuple[str, ...]
    epsilon: Number
    _a: int = field(init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.domain, (str, bytes)):
            raise TypeError('domain must be a sequence of strings')
        domain = tuple(self.domain)
        if len(domain) < 2 or any(not isinstance(x, str) for x in domain):
            raise ValueError('domain must contain at least two distinct strings')
        if len(set(domain)) != len(domain):
            raise ValueError('domain values must be unique')
        epsilon = privacy_budget(self.epsilon)
        object.__setattr__(self, 'domain', domain)
        object.__setattr__(self, 'epsilon', epsilon)
        object.__setattr__(self, '_a', _self_weight(epsilon))

    @property
    def denominator(self) -> int:
        return self._a + (len(self.domain) - 1) * _WEIGHT

    @property
    def likelihood_ratio(self) -> Fraction:
        """Exact largest ratio P(y|x)/P(y|x') for the implemented sampler."""
        return Fraction(self._a, _WEIGHT)

    def _index(self, value: str) -> int:
        if not isinstance(value, str):
            raise TypeError('value must be a string in the public domain')
        try:
            return self.domain.index(value)
        except ValueError as exc:
            # Do not include the private value in exception messages.
            raise ValueError('value is outside the public domain') from exc

    def probability(self, value: str, report: str) -> Fraction:
        """Exact transition probability, useful for exhaustive privacy audits."""
        x, y = self._index(value), self._index(report)
        return Fraction(self._a if x == y else _WEIGHT, self.denominator)

    def _decode_draw(self, index: int, draw: int) -> str:
        if not 0 <= index < len(self.domain) or not 0 <= draw < self.denominator:
            raise ValueError('Invalid sampler index or draw')
        if draw < self._a:
            return self.domain[index]
        other = (draw - self._a) // _WEIGHT
        return self.domain[other if other < index else other + 1]

    def randomize(self, value: str) -> str:
        index = self._index(value)
        return self._decode_draw(index, secrets.randbelow(self.denominator))

    def estimate_frequencies(self, reports: Iterable[str]) -> dict[str, Fraction]:
        """Unbiased category proportions using actual integer probabilities.

        Estimates may be negative or above one; no biased clipping is applied.
        Reports must use this exact public configuration and fresh randomness.
        """
        if self._a == _WEIGHT:
            raise ValueError('Uniform reports contain no frequency information')
        counts = dict.fromkeys(self.domain, 0)
        n = 0
        for report in reports:
            self._index(report)
            counts[report] += 1
            n += 1
        if not n:
            raise ValueError('At least one report is required')
        return {value: Fraction(count * self.denominator - n * _WEIGHT,
                                n * (self._a - _WEIGHT))
                for value, count in counts.items()}


@dataclass(frozen=True)
class BoundedNumeric:
    """Clip to public bounds, quantize to a public grid, then randomize.

    Reports and probability calculations are exact Fractions. The estimator
    targets the clipped, quantized population mean, not the unbounded raw mean.
    """

    lower: Number
    upper: Number
    bins: int
    epsilon: Number
    _rr: RandomizedResponse = field(init=False, repr=False)

    def __post_init__(self):
        lower, upper = as_fraction(self.lower, 'lower'), as_fraction(self.upper, 'upper')
        if lower >= upper:
            raise ValueError('lower must be less than upper')
        if isinstance(self.bins, bool) or not isinstance(self.bins, int) or not 2 <= self.bins <= 100_000:
            raise ValueError('bins must be an integer in [2, 100000]')
        rr = RandomizedResponse(tuple(str(i) for i in range(self.bins)), self.epsilon)
        object.__setattr__(self, 'lower', lower)
        object.__setattr__(self, 'upper', upper)
        object.__setattr__(self, 'epsilon', rr.epsilon)
        object.__setattr__(self, '_rr', rr)

    def _bin(self, value: Number) -> int:
        clipped = min(self.upper, max(self.lower, as_fraction(value, 'value')))
        position = (clipped - self.lower) * (self.bins - 1) / (self.upper - self.lower)
        # Nearest grid point, exact ties rounded upward.
        return (position + Fraction(1, 2)).numerator // (position + Fraction(1, 2)).denominator

    def _point(self, index: int) -> Fraction:
        return self.lower + (self.upper - self.lower) * Fraction(index, self.bins - 1)

    def _report_index(self, report: Number) -> int:
        point = as_fraction(report, 'report')
        index = self._bin(point)
        if self._point(index) != point:
            raise ValueError('report must be an exact public grid point')
        return index

    def quantize(self, value: Number) -> Fraction:
        return self._point(self._bin(value))

    def randomize(self, value: Number) -> Fraction:
        return self._point(int(self._rr.randomize(str(self._bin(value)))))

    def probability(self, value: Number, report: Number) -> Fraction:
        return self._rr.probability(str(self._bin(value)), str(self._report_index(report)))

    def estimate_mean(self, reports: Iterable[Number]) -> Fraction:
        frequencies = self._rr.estimate_frequencies(str(self._report_index(r)) for r in reports)
        return sum((self._point(int(index)) * p for index, p in frequencies.items()), Fraction())


@dataclass(frozen=True)
class TextRandomizer:
    """Fixed-size whitespace-token reports over a fixed public vocabulary.

    Unknown tokens map to <UNK>; text is truncated/padded to max_tokens. Every
    slot, including padding, is randomized with epsilon/max_tokens. No raw
    words, lengths, entity labels, or offsets are included in the report.
    """

    vocabulary: tuple[str, ...]
    max_tokens: int
    epsilon: Number
    _rr: RandomizedResponse = field(init=False, repr=False)
    PAD = '<PAD>'
    UNKNOWN = '<UNK>'

    def __post_init__(self):
        if isinstance(self.vocabulary, (str, bytes)):
            raise TypeError('vocabulary must be a sequence of public tokens')
        vocab = tuple(self.vocabulary)
        if (not vocab or any(not isinstance(t, str) or not t or len(t.split()) != 1 or t.strip() != t
                             or t in {self.PAD, self.UNKNOWN} for t in vocab)):
            raise ValueError('vocabulary must contain non-empty tokens without whitespace or reserved markers')
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError('max_tokens must be a positive integer')
        epsilon = privacy_budget(self.epsilon)
        rr = RandomizedResponse(vocab + (self.PAD, self.UNKNOWN), epsilon / self.max_tokens)
        object.__setattr__(self, 'vocabulary', vocab)
        object.__setattr__(self, 'epsilon', epsilon)
        object.__setattr__(self, '_rr', rr)

    def _encode(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str):
            raise TypeError('text must be a string')
        tokens = text.split()[:self.max_tokens]
        known = set(self.vocabulary)
        values = [token if token in known else self.UNKNOWN for token in tokens]
        return tuple(values + [self.PAD] * (self.max_tokens - len(values)))

    def randomize(self, text: str) -> tuple[str, ...]:
        return tuple(self._rr.randomize(token) for token in self._encode(text))

    def probability(self, text: str, report: Iterable[str]) -> Fraction:
        values = tuple(report)
        if len(values) != self.max_tokens:
            return Fraction(0)
        probability = Fraction(1)
        for token, output in zip(self._encode(text), values):
            probability *= self._rr.probability(token, output)
        return probability

    def randomize_text(self, text: str) -> str:
        """Keep report markers, including padding, for a fixed-size text report."""
        return ' '.join(self.randomize(text))
