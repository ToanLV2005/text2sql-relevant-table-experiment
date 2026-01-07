"""
get_final.py

Pipeline step (5):
  - Take top-k candidate tables from vector search
  - Render a structured prompt (Vietnamese)
  - Ask LLM to select the final tables
  - Validate & filter LLM output to avoid hallucinations / SQL output
"""

import os
import json
import re
import requests
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

try:
    from .get_candidate_e5 import get_candidate_pref as get_candidate_tables
    from .instructions.system_prompt_vi import SYSTEM_INSTRUCTION as SYSTEM_VI
    from .instructions.system_prompt_en import SYSTEM_INSTRUCTION as SYSTEM_EN
except ImportError:
    from get_candidate_e5 import get_candidate_pref as get_candidate_tables
    from instructions.system_prompt_vi import SYSTEM_INSTRUCTION as SYSTEM_VI
    from instructions.system_prompt_en import SYSTEM_INSTRUCTION as SYSTEM_EN

load_dotenv()
BASE_DIR = os.path.dirname(__file__)


# Config from env
TOP_K = int(os.getenv("TOP_K", "10"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://mkp-api.fptcloud.com/v1")
LLM_MODEL = os.getenv("TABLE_SELECTION_LLM_MODEL", "gpt-oss-20b")
print(f"Using {LLM_MODEL} for table selection")
print(f"USE_INSTRUCTION: {os.getenv('USE_INSTRUCTION', 'true')}")
LLM_API_KEY = os.getenv("LLM_API_KEY")
INSTRUCTION_LAN = os.getenv("INSTRUCTION_LAN","en")
USE_INSTRUCTION = os.getenv("USE_INSTRUCTION", "true").lower() == "true"


# Determine template path based on INSTRUCTION_LAN
if INSTRUCTION_LAN == "en":
    DEFAULT_TEMPLATE_NAME = "table_selection_template_en.txt"
else:
    DEFAULT_TEMPLATE_NAME = "table_selection_template_vi.txt"

TEMPLATE_PATH = os.getenv(
    "TABLE_SELECTION_TEMPLATE_PATH",
    os.path.join(BASE_DIR, "instructions", DEFAULT_TEMPLATE_NAME)
)

TABLE_USAGE_INSTRUCTIONS_PATH = os.getenv(
    "TABLE_INSTRUCTION_PATH",
    os.path.join(BASE_DIR, "instructions", "instruction.json")
)



# Regex helpers
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SQL_HINT_RE = re.compile(
    r"\b(select|from|join|where|group\s+by|order\s+by|create|view|insert|update|delete|with)\b",
    re.IGNORECASE,
)
DESC_RE = re.compile(r"^\s*Description:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
TABLE_SPLIT_RE = re.compile(r"^\s*===\s*TABLE\s*===\s*$", re.IGNORECASE | re.MULTILINE)
PK_LINE_RE = re.compile(r"^\s*-\s*Primary\s+Key\s*:\s*(.+?)\s*$", re.IGNORECASE)
FK_LINE_RE = re.compile(r"^\s*-\s*Foreign\s+Key\s*:\s*(.+?)\s*$", re.IGNORECASE)

# Utilities
def normalize_table_name(t: str) -> str:
    """Normalize table names returned by LLM."""
    if not t:
        return ""
    t = str(t).strip().strip("`'\"")
    t = re.sub(r"^\s*table\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def looks_like_sql(s: str) -> bool:
    """Help reject SQL or long hallucinated outputs."""
    s2 = (s or "").strip()
    if "\n" in s2 or len(s2) > 80:
        return True
    return bool(SQL_HINT_RE.search(s2))

def extract_json_object(text: str) -> Optional[dict]:
    """JSON extraction from messy LLM output."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None

def validate_and_filter_final_tables(
    result: dict,
    allowed_tables_in_order: List[str],
) -> List[str]:
    """
    Enforces:
      - valid table_name
      - no SQL-like strings
      - must exist in candidate tables
      - stable order, no duplicates
    """

    allowed_tables = {normalize_table_name(t): t for t in allowed_tables_in_order}

    raw = result.get("final_tables", [])
    if not isinstance(raw, list):
        return []

    picked = []
    for x in raw:
        # Ensure valid table name
        s = normalize_table_name(str(x))
        # Ensure no SQL
        if looks_like_sql(s):
            continue
        # Remove any indentation
        if not IDENT_RE.match(s):
            continue
        # Make sure the table is in candidate list
        if s in allowed_tables:
            picked.append(allowed_tables[s])

    # Remove duplicate
    seen, out = set(), []
    for t in picked:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

# Load template and instrction
def load_table_usage_instructions() -> Dict[str, List[str]]:
    """Load JSON containing use cases and the tables required."""
    path = TABLE_USAGE_INSTRUCTIONS_PATH
    if not path or not os.path.exists(path):
        return {}
    
    # Open JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        return {}
    
    # Convert json object into a dict of instructions {<instruction>: [table1, table2,...]}
    cleaned = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, list):
            tables = [normalize_table_name(x) for x in v if x]
            if tables:
                cleaned[k.strip()] = tables
    return cleaned

def extract_description_from_doc(doc: str) -> str:
    """Extract Description field from schema doc."""
    if not doc:
        return ""
    m = DESC_RE.search(doc)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in doc.splitlines() if ln.strip()]
    return lines[1][:240] if len(lines) > 1 else (lines[0][:240] if lines else "")


def extract_pk_fk_from_doc(doc: str) -> str:
    """
    Extract only Primary Key / Foreign Key lines from schema doc.
    """
    if not doc:
        return "(Không có PK/FK)"

    parts = TABLE_SPLIT_RE.split(doc)
    doc1 = parts[0] if parts else doc

    lines = [ln.rstrip() for ln in doc1.splitlines()]
    in_keys = False
    out = []

    for ln in lines:
        if not in_keys:
            if ln.strip().lower() == "keys:":
                in_keys = True
            continue

        if ln.strip().startswith(("Schema:", "=== TABLE ===")):
            break

        m_pk = PK_LINE_RE.match(ln)
        if m_pk:
            out.append(f"- Primary Key: {m_pk.group(1).strip()}")
            continue

        m_fk = FK_LINE_RE.match(ln)
        if m_fk:
            out.append(f"- Foreign Key: {m_fk.group(1).strip()}")
            continue

    return "\n".join(out) if out else "(Không có PK/FK)"

def load_template_text() -> str:
    """Load template for selection query"""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()
    
def render_template_vi(
    template_text: str,
    query: str,
    tables: List[Dict[str, str]],
    table_usage_instructions: Dict[str, List[str]],
) -> str:
    from jinja2 import Template
    return Template(template_text).render(
        query=query,
        tables=tables,
        table_usage_instructions=table_usage_instructions,
    )
    

# LLM call
def call_llm(system_prompt: str, user_prompt: str) -> dict:
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is missing")
    
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "top_p": 1,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    resp = r.json()

    msg = resp["choices"][0]["message"]
    content = msg.get("content", "")
    reasoning = msg.get("reasoning_content", "")

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = extract_json_object(content) or {"raw": content}

    result["reasoning_content"] = reasoning

    # Extract token usage
    usage = resp.get("usage", {})
    result["token_usage"] = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    return result


# Main entry
def get_final_tables(query: str, top_k: int = TOP_K) -> dict:
    cand = get_candidate_tables(query, top_k=top_k)
    candidates = cand.get("candidates", [])

    allowed_tables = [c["table"] for c in candidates if c.get("table")]

    tables_for_template = []
    for c in candidates:
        doc = c.get("doc", "") or ""
        tables_for_template.append({
            "name": c["table"],
            "description": extract_description_from_doc(doc),
            "keys": extract_pk_fk_from_doc(doc), 
        })

    # Load instructions only if USE_INSTRUCTION is enabled
    table_usage_instructions = load_table_usage_instructions() if USE_INSTRUCTION else {}

    user_prompt = render_template_vi(
        template_text=load_template_text(),
        query=query,
        tables=tables_for_template,
        table_usage_instructions=table_usage_instructions,
    )
    

    # Select system prompt based on INSTRUCTION_LAN
    system_prompt = SYSTEM_EN if INSTRUCTION_LAN == "en" else SYSTEM_VI

    result = call_llm(system_prompt, user_prompt)
    final_tables = validate_and_filter_final_tables(result, allowed_tables)

    if not final_tables and allowed_tables:
        final_tables = allowed_tables[: min(3, len(allowed_tables))]

    return {
        "query": query,
        "final_tables": final_tables,
        "reasoning_content": result.get("reasoning_content", ""),
        "token_usage": result.get("token_usage", {}),
    }

if __name__ == "__main__":
    q = input("Query: ").strip()
    print(json.dumps(get_final_tables(q), ensure_ascii=False, indent=2))
