"""Retrieval and privacy research pipelines, without import-time model loading."""
from .rag import HybridRAG, NormalRAG
from .mechanisms import RandomizedResponse, BoundedNumeric, TextRandomizer
from .accounting import BudgetAccountant, BudgetExceededError
from .private_rag import LocalDPRAG

__version__ = "0.2.0"
__all__ = ["NormalRAG", "HybridRAG", "LPRAG", "LDPRAG", "PrivacyGateRAG",
           "RandomizedResponse", "BoundedNumeric", "TextRandomizer",
           "BudgetAccountant", "BudgetExceededError", "LocalDPRAG"]


def __getattr__(name):
    if name in {"LPRAG", "LDPRAG"}:
        from .ldp import LPRAG
        return LPRAG
    if name == "PrivacyGateRAG":
        from .privacy_gate import PrivacyGateRAG
        return PrivacyGateRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
