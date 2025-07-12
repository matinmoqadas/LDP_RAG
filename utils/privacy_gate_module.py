import json
import re
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# -------------- Configuration --------------
base_url   = "https://api.avalai.ir/v1"
api_key    = "aa-"
model_name = "gpt-4o-mini"

llm = ChatOpenAI(
    base_url=base_url,
    model=model_name,
    api_key=api_key,
    temperature=0.0,
)

MAX_TOKENS = 512

# -------------- Prompt Template --------------
PRIVACY_GATE_TEMPLATE = """
You are the Privacy Gate.

Your job: rewrite the provided text chunk to obscure details per vagueness ε in [0.1,1.0].
**Output**: ONLY a single JSON object, no markdown, no code fences, no explanations.

Vagueness levels:
 • ε=1.0 → minimal vagueness (strip only explicit secrets)
 • ε=0.7 → slight vagueness (generalize minor specifics)
 • ε=0.5 → moderate vagueness (remove most specifics, keep core meaning)
 • ε=0.3 → high vagueness (only broad actions remain)
 • ε=0.1 → maximal vagueness (bare outline)

Label: {label}
Epsilon: {epsilon}
Original Chunk:
\"\"\"{chunk}\"\"\"

Return exactly:
{{ "rewritten": "<sanitized text>" }}
"""

def privacy_gate_sanitize(
    chunk: str,
    label: Literal["PUBLIC", "SENSITIVE", "CONFIDENTIAL"],
    epsilon: float
) -> str:
    prompt = PRIVACY_GATE_TEMPLATE.format(
        label=label,
        epsilon=epsilon,
        chunk=chunk.replace('"','\\"')
    )

    messages = [
        SystemMessage(content="You are a privacy-focused assistant."),
        HumanMessage(content=prompt)
    ]
    resp = llm(messages, max_tokens=MAX_TOKENS)
    raw = resp.content.strip()

    # Try to pull out the first {...} block
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        # Dump raw for debugging
        raise ValueError(f"LLM response contained no JSON.\n\n---RAW---\n{raw}\n---END RAW---")

    json_str = match.group(0)
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON:\n{json_str}\nError: {e}")

    if "rewritten" not in obj:
        raise ValueError(f"JSON has no 'rewritten' key:\n{obj}")

    return obj["rewritten"]
