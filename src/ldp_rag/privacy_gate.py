"""Prompt-driven privacy rewriting. Epsilon here controls wording, not DP."""
import json
import math
import re
from .providers import invoke_text
from .rag import NormalRAG

PRECISION_TMPL = '''You are the Privacy Gate.
Rewrite the text so its vagueness matches epsilon={epsilon}.
1.0: keep almost all specifics; 0.7: generalise a little; 0.5: blur most
specifics; 0.3: broad actions only; 0.1: bare outline. Keep the core meaning.
Label: {label}
Original text:
{chunk}
Return JSON only: {{"rewritten": "<sanitised>"}}'''

DEEP_TMPL = '''Rewrite the input to be even vaguer: shorten sentences, replace
specifics with generic terms, change wording and keep intent.
Input:
{chunk}
Return JSON only: {{"rewritten": "<even more vague text>"}}'''


def _call(llm, prompt):
    if llm is None:
        raise ValueError("A rewriting LLM is required")
    response = invoke_text(llm, prompt, "You are a privacy-focused assistant.")
    response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response).strip()
    obj = json.loads(response)
    if not isinstance(obj, dict) or not isinstance(obj.get("rewritten"), str) or not obj["rewritten"].strip():
        raise ValueError("Privacy Gate response must contain non-empty string 'rewritten'")
    return obj["rewritten"]


def sanitise_once(chunk, label, epsilon, *, llm):
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("Privacy Gate epsilon must be in (0, 1]")
    if label not in {"PUBLIC", "SENSITIVE", "CONFIDENTIAL"}:
        raise ValueError("Invalid privacy label")
    return _call(llm, PRECISION_TMPL.format(chunk=chunk, label=label, epsilon=epsilon))


def deep_obfuscate(previous, *, llm):
    return _call(llm, DEEP_TMPL.format(chunk=previous))


def privacy_gate_pipeline(text, label, eps_schedule=(1.0, 0.7, 0.5, 0.3, 0.1), deep_rounds=4, *, llm, deep_llm=None):
    schedule = list(eps_schedule)
    if not schedule or len(set(schedule)) != len(schedule) or schedule != sorted(schedule, reverse=True):
        raise ValueError("eps_schedule must be non-empty, unique, and descending")
    if any(not math.isfinite(eps) or not 0 < eps <= 1 for eps in schedule):
        raise ValueError("Privacy Gate epsilon must be in (0, 1]")
    if not isinstance(deep_rounds, int) or isinstance(deep_rounds, bool) or deep_rounds < 0:
        raise ValueError("deep_rounds must be a non-negative integer")
    results, current = {}, text
    for eps in schedule:
        current = sanitise_once(current, label, eps, llm=llm)
        results[eps] = [current]
    for _ in range(deep_rounds):
        current = deep_obfuscate(current, llm=deep_llm if deep_llm is not None else llm)
        results[schedule[-1]].append(current)
    return results


class PrivacyGateRAG(NormalRAG):
    """Rewrite retrieved context before answer generation using an explicit gate."""
    def __init__(self, *args, gate_llm=None, epsilon=0.3, label="SENSITIVE", **kwargs):
        if gate_llm is None:
            raise ValueError("gate_llm must be supplied explicitly; it receives raw context")
        if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
            raise ValueError("Privacy Gate epsilon must be in (0, 1]")
        if label not in {"PUBLIC", "SENSITIVE", "CONFIDENTIAL"}:
            raise ValueError("Invalid privacy label")
        super().__init__(*args, **kwargs)
        self.gate_llm, self.epsilon, self.label = gate_llm, epsilon, label

    def retrieve_context(self, question, top_k=4):
        return [sanitise_once(chunk, self.label, self.epsilon, llm=self.gate_llm)
                for chunk in super().retrieve_context(question, top_k)]
