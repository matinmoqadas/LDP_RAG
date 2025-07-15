# RAG‑Pipeline

A flexible **Retrieval‑Augmented Generation (RAG) evaluation harness** supporting multiple back‑ends, adaptive chunking, and pluggable metrics — all driven from a single `main.py`.

---

## 1 · Project layout

```text
project/
│
├─ models/                   # each file defines ONE RAG class
│   ├─ normal_rag.py         # class NormalRAG
│   ├─ privacy_gate_rag.py   # class PrivacyGateRAG
│   ├─ ldp_rag.py            # class LDPRAG
│   └─ adaptive_rag.py       # class AdaptiveRAG
│
├─ utils/
│   ├─ metrics.py            # faithfulness(), answer_relevancy(), …
│   └─ adaptive_chunking.py  # AdaptiveChunkSplitter (or get_splitter())
│
├─ main.py                   # CLI runner
└─ README.md                 # this file
```

```text
datasets/
└─ dataset-ldp/
    └─ Data_LDP/
        ├─ Docs/                       # knowledge base (.docx files)
        ├─ Attack Question/            # prompts to ask the RAG
        │   ├─ Customer Service/1.json … 10.json
        │   ├─ E‑commerce/…
        │   └─ … (Healthcare, Finance, …)
        └─ MetaDatas/
```

---

## 2 · Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip wheel

# Core libs
pip install langchain faiss-cpu openai google-generativeai json5

# Optional: pandas for inspecting output
pip install pandas
```

> **Python ≥ 3.9** is recommended (FAISS wheels available from 3.9 up).

---

## 3 · Environment variables

| Variable         | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| `OPENAI_API_KEY` | API key for GPT‑style back‑ends (AvalAI proxy) |
| `GOOGLE_API_KEY` | Gemini back‑end (optional)                     |

You can also pass `--api_key` and `--base_url` directly on the CLI.

---

## 4 · Quick start

```bash
python main.py \
    --rag_type normal \
    --metrics faithfulness answer_relevancy \
    --adaptive_chunking \
    --output_path ./runs/demo.jsonl
```

This will:

1. Build / load a FAISS index from `datasets/dataset-ldp/Data_LDP/Docs`.
2. Recursively scan every `*.json` inside `Attack Question/…`.
3. For each question:

   * retrieve `top_k` chunks (default 4)
   * generate an answer with the selected RAG back‑end
   * evaluate requested metrics
   * append one line to `runs/demo.jsonl`.

---

## 5 · CLI reference

| Flag                  | Default                                         | Description                                                                  |
| --------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------- |
| `--rag_type`          | —                                               | `normal`, `privacy_gate`, `ldp`, `adaptive` (maps to a class in **models/**) |
| `--docs_dir`          | `datasets/dataset-ldp/Data_LDP/Docs`            | Folder with `.docx` knowledge docs                                           |
| `--questions_dir`     | `datasets/dataset-ldp/Data_LDP/Attack Question` | Root of attack‑question tree                                                 |
| `--metrics`           | —                                               | Space‑separated list of functions in `utils.metrics`                         |
| `--adaptive_chunking` | `false`                                         | Use `utils.adaptive_chunking.AdaptiveChunkSplitter`                          |
| `--output_path`       | —                                               | Destination `.jsonl` file                                                    |
| `--index_name`        | `faiss_index`                                   | Sub‑folder in `RAG_Database/`                                                |
| `--top_k`             | `4`                                             | Retrieved context chunks                                                     |
| `--llm_name`          | `openai`                                        | `openai` or `gemini`                                                         |
| `--model_name`        | `gpt-4o-mini`                                   | Underlying chat model                                                        |
| `--api_key`           | env var or `API_KEY`                            | API key override                                                             |
| `--base_url`          | `https://api.avalai.ir/v1`                      | OpenAI‑compatible endpoint                                                   |

---

## 6 · Output format

Each **line** in the output `.jsonl` file is a record like:

```jsonc
{
  "question": "How would this medical detail impact treatment plans: …?",
  "question_type": "Healthcare",
  "label": "Sensitive",
  "source_sentence": "CKD increases the risk of cardiovascular disease, …",
  "answer": "First, clinicians would …",
  "metrics": {
    "faithfulness": 0.93,
    "answer_relevancy": 0.82
  },
  "file": "/…/Attack Question/Healthcare/7.json"
}
```

```python
import pandas as pd
df = pd.read_json("runs/demo.jsonl", lines=True)
```

---

## 7 · Extending the pipeline

### 7.1 Add a new RAG back‑end

1. Add a file in **models/**, expose one class with the same public API.
2. Register it in `RAG_CLASS_REGISTRY` inside `main.py`:

```python
RAG_CLASS_REGISTRY["my_rag"] = "models.my_rag:MyRAG"
```

### 7.2 Add a metric

1. Edit `utils/metrics.py`:

```python
def rouge_l(predicted: str, reference: str, context: list[str]) -> float:
    ...
```

2. Request it: `--metrics rouge_l`.

### 7.3 Custom chunker

Expose `AdaptiveChunkSplitter` (or a `get_splitter()` factory) in `utils/adaptive_chunking.py`, then enable with `--adaptive_chunking`.

---

## 8 · Troubleshooting

| Issue                        | Fix                                          |
| ---------------------------- | -------------------------------------------- |
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` (or `faiss-gpu`)     |
| 401 / 403 from LLM           | Check `OPENAI_API_KEY` and `--base_url`      |
| “No documents loaded yet”    | Ensure `.docx` files exist in `--docs_dir`   |
| Metric error `KeyError`      | Verify the metric name passed in `--metrics` |

---

## 9 · License

Replace this section with your project’s license.

---

Happy experimenting \:rocket:
