"""Exact sequential privacy accounting with atomic, durable reservations."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sqlite3
from threading import RLock

from .mechanisms import Number, privacy_budget


class BudgetExceededError(ValueError):
    """A release would exceed the configured budget for this privacy unit."""


class BudgetAccountant:
    """Track sum(epsilon) per locally assigned privacy unit using SQLite.

    Pass a persistent path to carry accounting across restarts/processes. An
    omitted path creates an in-memory ledger for this instance only. Protect
    the ledger and reuse stable unit IDs; deleting it or changing IDs resets
    accounting. Reservations commit before any output is released and are not
    refunded after errors. The ledger itself must stay within the trust boundary.
    """

    def __init__(self, epsilon_limit: Number, *, path: str | Path | None = None):
        self._epsilon_limit = privacy_budget(epsilon_limit)
        self._lock = RLock()
        self._db = sqlite3.connect(str(path) if path is not None else ':memory:',
                                   timeout=30, isolation_level=None, check_same_thread=False)
        try:
            self._db.execute('PRAGMA synchronous=FULL')
            self._db.execute('CREATE TABLE IF NOT EXISTS ldp_policy (id INTEGER PRIMARY KEY CHECK(id=1), epsilon_limit TEXT NOT NULL)')
            self._db.execute('CREATE TABLE IF NOT EXISTS ldp_spending (unit TEXT PRIMARY KEY, epsilon TEXT NOT NULL)')
            self._db.execute('BEGIN IMMEDIATE')
            row = self._db.execute('SELECT epsilon_limit FROM ldp_policy WHERE id=1').fetchone()
            if row is None:
                self._db.execute('INSERT INTO ldp_policy VALUES (1, ?)', (str(self._epsilon_limit),))
            elif Fraction(row[0]) != self._epsilon_limit:
                raise ValueError('Existing ledger has a different epsilon limit')
            self._db.execute('COMMIT')
        except BaseException:
            if self._db.in_transaction:
                self._db.execute('ROLLBACK')
            self._db.close()
            raise

    @property
    def epsilon_limit(self) -> Fraction:
        return self._epsilon_limit

    @staticmethod
    def _validate_unit(unit: str) -> None:
        if not isinstance(unit, str) or not unit:
            raise ValueError('privacy_unit must be a non-empty local identifier')

    def _read_spent(self, unit: str) -> Fraction:
        row = self._db.execute('SELECT epsilon FROM ldp_spending WHERE unit=?', (unit,)).fetchone()
        spent = Fraction(row[0]) if row else Fraction(0)
        if not 0 <= spent <= self.epsilon_limit:
            raise ValueError('Invalid spending in privacy ledger')
        return spent

    def spent(self, unit: str) -> Fraction:
        self._validate_unit(unit)
        with self._lock:
            return self._read_spent(unit)

    def remaining(self, unit: str) -> Fraction:
        return self.epsilon_limit - self.spent(unit)

    def reserve(self, unit: str, epsilon: Number) -> Fraction:
        """Atomically charge a release; return the new total. No refund API."""
        self._validate_unit(unit)
        cost = privacy_budget(epsilon)
        with self._lock:
            self._db.execute('BEGIN IMMEDIATE')
            try:
                total = self._read_spent(unit) + cost
                if total > self.epsilon_limit:
                    raise BudgetExceededError('Privacy budget exhausted for this unit')
                self._db.execute(
                    'INSERT INTO ldp_spending VALUES (?, ?) ON CONFLICT(unit) DO UPDATE SET epsilon=excluded.epsilon',
                    (unit, str(total)))
                self._db.execute('COMMIT')
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute('ROLLBACK')
                raise
        return total

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
