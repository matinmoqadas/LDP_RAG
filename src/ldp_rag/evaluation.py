"""Question dataset traversal and streaming JSONL evaluation."""
import json
from pathlib import Path


def iterate_question_files(questions_dir):
    root = Path(questions_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(root.rglob('*.json'))
    if not paths:
        raise ValueError(f"No question JSON files under {root}")
    for path in paths:
        text = path.read_text(encoding='utf-8')
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import json5
            except ImportError as exc:
                raise ValueError(f"Invalid JSON in {path}; install ldp-rag[json5] for JSON5 input") from exc
            data = json5.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of question objects in {path}")
        relative = path.relative_to(root)
        category = relative.parts[0] if len(relative.parts) > 1 else 'unknown'
        for index, row in enumerate(data):
            if not isinstance(row, dict) or not isinstance(row.get('question'), str) or not row['question'].strip():
                raise ValueError(f"Invalid question row {index} in {path}")
            yield {**row, 'type': row.get('type', category), '_file': str(path)}


def evaluate_questions(rag, rows, *, metrics=None, top_k=4):
    for row in rows:
        answer, context = rag.generate(row['question'], top_k=top_k, include_context=True)
        scores = {}
        for name, function in (metrics or {}).items():
            # Evaluation failures propagate; callers decide their error policy.
            scores[name] = function(predicted=answer, reference=row.get('reference'), context=context, question=row['question'])
        yield {'question': row['question'], 'question_type': row.get('type', 'unknown'),
               'label': row.get('label'), 'source_sentence': row.get('source_sentence'),
               'answer': answer, 'metrics': scores, 'file': row.get('_file')}


def run_evaluation(rag, questions_dir, output_path, *, metrics=None, top_k=4):
    path = Path(output_path)
    root = Path(questions_dir)
    if path.resolve() in {p.resolve() for p in root.rglob('*.json')}:
        raise ValueError('Output must not overwrite a question input')
    rows = iterate_question_files(root)
    # Validate the first row before opening the output file.
    first = next(rows, None)
    if first is None:
        raise ValueError('Question dataset is empty')
    from itertools import chain
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as output:
        for result in evaluate_questions(rag, chain([first], rows), metrics=metrics, top_k=top_k):
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            output.flush()
            count += 1
    return count
