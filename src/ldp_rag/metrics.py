"""Evaluation helpers. No clients or NLP models are initialized on import."""
import re
from typing import Dict, Set

ATOM_WEIGHTS = {
    "email": 5.0, "phone": 5.0, "id": 5.0, "address": 4.0,
    "name": 3.0, "date": 2.0, "default": 1.0,
}

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
DATE_RE  = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
ID_RE    = re.compile(r"\b\d{4,}\b")

def _detect_atom_type(token: str) -> str:
    if EMAIL_RE.search(token): return "email"
    if PHONE_RE.search(token): return "phone"
    if DATE_RE.search(token):  return "date"
    if ID_RE.search(token):    return "id"
    return "default"

def _ensure_nlp():
    try:
        import spacy  # lazy import
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(f"Failed to load spacy model 'en_core_web_sm'. Please install it with: python -m spacy download en_core_web_sm") from e

_STOP = {
    "the","a","an","and","or","of","to","in","for","on","with","by","at","is","are",
}

def _extract_atoms(txt: str, nlp, min_len: int = 2) -> Set[str]:
    ents = {e.text.strip().lower() for e in nlp(txt).ents if e.text.strip()}
    toks = {
        t.lower() for t in re.findall(r"[A-Za-z0-9@._+-]+", txt)
        if len(t) >= min_len and t.lower() not in _STOP
    }
    return ents | toks

def cold_stats_leakage(reference_answer: str, generated_answer: str, nlp=None) -> Dict[str, float]:
    if not reference_answer or not generated_answer:
        return {"leak_rate": 0.0, "ils": 1.0}
    
    if nlp is None:
        nlp = _ensure_nlp()
    priv_atoms = _extract_atoms(reference_answer, nlp)
    pub_atoms  = _extract_atoms(generated_answer, nlp)

    weighted_total = 0.0
    weighted_overlap = 0.0
    for atom in priv_atoms:
        w = ATOM_WEIGHTS.get(_detect_atom_type(atom), ATOM_WEIGHTS["default"])
        weighted_total += w
        if atom in pub_atoms:
            weighted_overlap += w

    leak_rate = weighted_overlap / max(1e-9, weighted_total)
    return {"leak_rate": float(leak_rate), "ils": float(1.0 - leak_rate)}



def evaluate_bleu(reference, candidate):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    refs = [reference] if isinstance(reference, str) else reference
    if not refs or not candidate:
        return 0.0
    return sentence_bleu([ref.split() for ref in refs], candidate.split(), smoothing_function=SmoothingFunction().method1)


def evaluate_rouge(reference, candidate):
    from rouge import Rouge
    if not reference or not candidate:
        return {name: {key: 0.0 for key in ('p', 'r', 'f')} for name in ('rouge-1', 'rouge-2', 'rouge-l')}
    return Rouge().get_scores(candidate, reference)[0]


def bleu(*, predicted, reference, context=None, question=None):
    if reference is None:
        raise ValueError("BLEU requires a reference")
    return evaluate_bleu(reference, predicted)


def rouge(*, predicted, reference, context=None, question=None):
    if reference is None:
        raise ValueError("ROUGE requires a reference")
    return evaluate_rouge(reference, predicted)


def evaluate_ragas(user_input, retrieved_contexts, generated_answer, reference_answer=None, *, llm, embeddings):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy
    data = {'question': [user_input], 'contexts': [retrieved_contexts], 'answer': [generated_answer]}
    if reference_answer is not None:
        data['ground_truth'] = [reference_answer]
    result = evaluate(dataset=Dataset.from_dict(data), metrics=[Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)], llm=llm, embeddings=embeddings, show_progress=False)
    return result.to_pandas().iloc[0].to_dict()


def make_ragas_metrics(*, llm, embeddings):
    """Return evaluation callbacks with the same signature as pipeline metrics."""
    def metric(name):
        def run(*, predicted, reference=None, context=None, question=None):
            return evaluate_ragas(question, context or [], predicted, reference, llm=llm, embeddings=embeddings)[name]
        return run
    return {name: metric(name) for name in ('faithfulness', 'answer_relevancy')}
