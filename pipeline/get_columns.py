"""
get_columns.py

Pipeline step (7):
  - Take final tables from table selection (step 6)
  - Match table names back to their full schema (with columns)
  - Ask LLM to select the relevant columns
  - Validate & filter LLM output
"""

import os
import json
import re
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

try:
    from .get_final import get_final_tables
    from .instructions.system_prompt_column_vi import SYSTEM_INSTRUCTION as SYSTEM_COL_VI
    from .instructions.system_prompt_column_en import SYSTEM_INSTRUCTION as SYSTEM_COL_EN
except ImportError:
    from get_final import get_final_tables
    from instructions.system_prompt_column_vi import SYSTEM_INSTRUCTION as SYSTEM_COL_VI
    from instructions.system_prompt_column_en import SYSTEM_INSTRUCTION as SYSTEM_COL_EN

load_dotenv()
BASE_DIR = os.path.dirname(__file__)

# Config from env
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://mkp-api.fptcloud.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-120b")
LLM_API_KEY = os.getenv("LLM_API_KEY")
INSTRUCTION_LAN = os.getenv("INSTRUCTION_LAN", "en")
SCHEMA_TXT_FILE = os.getenv("SCHEMA_TXT_FILE", "vi_schema.txt")

# Determine template path based on INSTRUCTION_LAN
if INSTRUCTION_LAN == "en":
    DEFAULT_COL_TEMPLATE_NAME = "column_selection_template_en.txt"
else:
    DEFAULT_COL_TEMPLATE_NAME = "column_selection_template_vi.txt"

COL_TEMPLATE_PATH = os.getenv(
    "COLUMN_SELECTION_TEMPLATE_PATH",
    os.path.join(BASE_DIR, "instructions", DEFAULT_COL_TEMPLATE_NAME)
)

# Regex patterns
TABLE_SPLIT_RE = re.compile(r"^\s*===\s*TABLE\s*===\s*$", re.IGNORECASE | re.MULTILINE)
DESC_RE = re.compile(r"^\s*Description:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
SCHEMA_START_RE = re.compile(r"^\s*Schema:\s*$", re.IGNORECASE | re.MULTILINE)
COLUMN_RE = re.compile(r"^\s*-\s*(\w+)\s+(\w+.*?)\s*$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Pattern: "Foreign Key: column_name -> table.column"
FK_RE = re.compile(r"Foreign Key:\s*(\w+)\s*->\s*(\w+)\.(\w+)", re.IGNORECASE)
PK_RE = re.compile(r"Primary Key:\s*(\w+)", re.IGNORECASE)


def load_schema_file() -> str:
    """Load the full schema file."""
    schema_path = os.path.join(BASE_DIR, SCHEMA_TXT_FILE)
    with open(schema_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_all_tables(schema_text: str) -> Dict[str, Dict]:
    """
    Parse schema text into a dict of tables with their full info.
    Returns: {table_name: {"description": ..., "columns": [...], "keys": ..., "raw": ...,
                           "primary_key": ..., "foreign_keys": [...]}}
    """
    tables = {}
    parts = TABLE_SPLIT_RE.split(schema_text)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract table name
        lines = part.splitlines()
        table_name = None
        for line in lines:
            if line.strip().lower().startswith("table:"):
                table_name = line.split(":", 1)[1].strip()
                break

        if not table_name:
            continue

        # Extract description
        desc_match = DESC_RE.search(part)
        description = desc_match.group(1).strip() if desc_match else ""

        # Extract columns
        columns = []
        in_schema = False
        in_keys = False
        keys_lines = []
        primary_key = None
        foreign_keys = []  # List of {"column": ..., "ref_table": ..., "ref_column": ...}

        for line in lines:
            stripped = line.strip()

            if stripped.lower() == "schema:":
                in_schema = True
                in_keys = False
                continue
            elif stripped.lower() == "keys:":
                in_schema = False
                in_keys = True
                continue

            if in_schema:
                col_match = COLUMN_RE.match(line)
                if col_match:
                    col_name = col_match.group(1)
                    col_type = col_match.group(2).strip()
                    columns.append({"name": col_name, "type": col_type})

            if in_keys and stripped.startswith("-"):
                keys_lines.append(stripped)
                # Parse Primary Key
                pk_match = PK_RE.search(stripped)
                if pk_match:
                    primary_key = pk_match.group(1)
                # Parse Foreign Key
                fk_match = FK_RE.search(stripped)
                if fk_match:
                    foreign_keys.append({
                        "column": fk_match.group(1),
                        "ref_table": fk_match.group(2),
                        "ref_column": fk_match.group(3),
                    })

        tables[table_name] = {
            "description": description,
            "columns": columns,
            "keys": "\n".join(keys_lines) if keys_lines else "(No keys)",
            "raw": part,
            "primary_key": primary_key,
            "foreign_keys": foreign_keys,
        }

    return tables


def get_tables_with_columns(table_names: List[str], all_tables: Dict[str, Dict]) -> List[Dict]:
    """
    Given a list of table names, return their full schema info for the template.
    """
    result = []
    for name in table_names:
        if name in all_tables:
            info = all_tables[name]
            result.append({
                "name": name,
                "description": info["description"],
                "columns": info["columns"],
                "keys": info["keys"],
            })
    return result


def load_column_template() -> str:
    """Load the column selection template."""
    with open(COL_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def render_column_template(
    template_text: str,
    query: str,
    tables: List[Dict],
) -> str:
    """Render the column selection template with Jinja2."""
    from jinja2 import Template
    return Template(template_text).render(
        query=query,
        tables=tables,
    )


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


def call_llm(system_prompt: str, user_prompt: str) -> dict:
    """Call the LLM API."""
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
        "max_tokens": 2048,
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


def validate_columns(
    result: dict,
    allowed_tables: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Validate LLM output:
    - Only keep tables that exist in allowed_tables
    - Only keep columns that exist in each table
    - Remove duplicates

    Supports both old format {"tables": {...}} and new format {"results": [...]}
    """
    validated = {}

    # Try new format first: {"results": [...]}
    raw_results = result.get("results", [])
    if isinstance(raw_results, list) and raw_results:
        for table_entry in raw_results:
            if not isinstance(table_entry, dict):
                continue

            table_name = table_entry.get("table_name", "").strip()
            columns = table_entry.get("columns", [])

            # Check if table is allowed
            if table_name not in allowed_tables:
                continue

            # Get valid columns for this table
            valid_cols = set(allowed_tables[table_name])

            # Filter columns
            if not isinstance(columns, list):
                continue

            filtered_cols = []
            seen = set()
            for col in columns:
                col = str(col).strip()
                if col in valid_cols and col not in seen:
                    filtered_cols.append(col)
                    seen.add(col)

            if filtered_cols:
                validated[table_name] = filtered_cols

        return validated

    # Fallback to old format: {"tables": {...}}
    raw_tables = result.get("tables", {})
    if not isinstance(raw_tables, dict):
        return {}

    for table_name, columns in raw_tables.items():
        # Normalize table name
        table_name = table_name.strip()

        # Check if table is allowed
        if table_name not in allowed_tables:
            continue

        # Get valid columns for this table
        valid_cols = set(allowed_tables[table_name])

        # Filter columns
        if not isinstance(columns, list):
            continue

        filtered_cols = []
        seen = set()
        for col in columns:
            col = str(col).strip()
            if col in valid_cols and col not in seen:
                filtered_cols.append(col)
                seen.add(col)

        if filtered_cols:
            validated[table_name] = filtered_cols

    return validated


def add_missing_join_columns(
    selected_columns: Dict[str, List[str]],
    all_tables: Dict[str, Dict],
) -> Dict[str, List[str]]:
    """
    Auto-add missing FK/PK columns needed to join the selected tables.

    For each selected table:
    - If it has a FK pointing to another selected table, add the FK column
    - If another selected table has a FK pointing to this table's PK, add the PK column

    This ensures all JOIN paths are covered without adding unnecessary columns.
    """
    selected_table_names = set(selected_columns.keys())

    if len(selected_table_names) <= 1:
        # No joins needed for single table
        return selected_columns

    result = {table: list(cols) for table, cols in selected_columns.items()}

    for table_name in selected_table_names:
        table_info = all_tables.get(table_name, {})

        # Add FK columns if they point to another selected table
        for fk in table_info.get("foreign_keys", []):
            ref_table = fk["ref_table"]
            fk_column = fk["column"]
            ref_column = fk["ref_column"]

            # Only add if the referenced table is also selected
            if ref_table in selected_table_names:
                # Add FK column to current table if not already present
                if fk_column not in result[table_name]:
                    result[table_name].append(fk_column)

                # Add referenced PK column to the referenced table if not already present
                if ref_table in result and ref_column not in result[ref_table]:
                    result[ref_table].append(ref_column)

    return result


def get_final_columns(query: str, top_k: int = 10) -> dict:
    """
    Main entry point for column selection.

    1. Get final tables from table selection
    2. Load full schema and match tables
    3. Call LLM to select columns
    4. Validate and return results
    """
    # Step 1: Get final tables
    table_result = get_final_tables(query, top_k=top_k)
    final_tables = table_result.get("final_tables", [])

    if not final_tables:
        return {
            "query": query,
            "final_tables": [],
            "columns": {},
            "table_selection_reasoning": table_result.get("reasoning_content", ""),
            "column_selection_reasoning": "",
            "token_usage": {
                "table_selection": table_result.get("token_usage", {}),
                "column_selection": {},
            },
        }

    # Step 2: Load schema and get table info
    schema_text = load_schema_file()
    all_tables = parse_all_tables(schema_text)
    tables_with_columns = get_tables_with_columns(final_tables, all_tables)

    # Build allowed columns map for validation
    allowed_columns = {}
    for t in tables_with_columns:
        allowed_columns[t["name"]] = [c["name"] for c in t["columns"]]

    # Step 3: Render template and call LLM
    template_text = load_column_template()
    user_prompt = render_column_template(
        template_text=template_text,
        query=query,
        tables=tables_with_columns,
    )

    # Select system prompt
    system_prompt = SYSTEM_COL_EN if INSTRUCTION_LAN == "en" else SYSTEM_COL_VI

    result = call_llm(system_prompt, user_prompt)

    # Step 4: Validate columns
    validated_columns = validate_columns(result, allowed_columns)

    # Fallback: if no valid columns, include all columns from selected tables
    if not validated_columns:
        validated_columns = allowed_columns

    # Step 5: Auto-add missing JOIN columns (FK/PK) for selected tables
    validated_columns = add_missing_join_columns(validated_columns, all_tables)

    return {
        "query": query,
        "final_tables": final_tables,
        "columns": validated_columns,
        "table_selection_reasoning": table_result.get("reasoning_content", ""),
        "column_selection_reasoning": result.get("reasoning_content", ""),
        "token_usage": {
            "table_selection": table_result.get("token_usage", {}),
            "column_selection": result.get("token_usage", {}),
        },
    }


def get_columns_from_tables(query: str, final_tables: List[str]) -> dict:
    """
    Alternative entry point: provide tables directly instead of running table selection.
    Useful for testing column selection independently.
    """
    if not final_tables:
        return {
            "query": query,
            "final_tables": [],
            "columns": {},
            "reasoning_content": "",
            "token_usage": {},
        }

    # Load schema and get table info
    schema_text = load_schema_file()
    all_tables = parse_all_tables(schema_text)
    tables_with_columns = get_tables_with_columns(final_tables, all_tables)

    # Build allowed columns map
    allowed_columns = {}
    for t in tables_with_columns:
        allowed_columns[t["name"]] = [c["name"] for c in t["columns"]]

    # Render template and call LLM
    template_text = load_column_template()
    user_prompt = render_column_template(
        template_text=template_text,
        query=query,
        tables=tables_with_columns,
    )

    system_prompt = SYSTEM_COL_EN if INSTRUCTION_LAN == "en" else SYSTEM_COL_VI
    result = call_llm(system_prompt, user_prompt)

    # Validate columns
    validated_columns = validate_columns(result, allowed_columns)

    if not validated_columns:
        validated_columns = allowed_columns

    # Auto-add missing JOIN columns (FK/PK) for selected tables
    validated_columns = add_missing_join_columns(validated_columns, all_tables)

    return {
        "query": query,
        "final_tables": final_tables,
        "columns": validated_columns,
        "reasoning_content": result.get("reasoning_content", ""),
        "token_usage": result.get("token_usage", {}),
    }


if __name__ == "__main__":
    print(f"Using model: {LLM_MODEL}")
    print(f"Language: {INSTRUCTION_LAN}")
    print("-" * 50)

    q = input("Query: ").strip()
    result = get_final_columns(q)
    print(json.dumps(result, ensure_ascii=False, indent=2))
