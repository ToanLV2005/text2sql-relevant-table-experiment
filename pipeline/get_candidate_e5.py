"""
E5-based table retrieval 

2 modes:
  - get_candidate_pref()   : query-prefixed retrieval ("query: ...") 
  - get_candidate_nopref() : raw-query retrieval                     

Both functions return a normalized result:
  {
    "query": <query>,
    "candidates": [
        {"table": <table_name>, "score": <float>, "doc": <schema_doc>}
    ]
  }
"""

import os
import re
import json
from typing import List, Dict, Any

import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Config
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION_E5", "vi_schema_E5")
MODEL_NAME = os.getenv("EMBED_MODEL_E5", "intfloat/multilingual-e5-large")
TOP_K = int(os.getenv("TOP_K", "10"))

PREF_VECTOR_NAME = os.getenv("PREF_VECTOR_NAME", "pref")
NOPREF_VECTOR_NAME = os.getenv("NOPREF_VECTOR_NAME", "nopref")

PAYLOAD_DOC_KEY = os.getenv("PAYLOAD_DOC_KEY", "doc")


# Init model
model = SentenceTransformer(MODEL_NAME)

# Regex to extract table name
TABLE_LINE_RE = re.compile(r"^\s*Table:\s*(.+?)\s*$", re.IGNORECASE)


# Helpers
def extract_table_name(doc: str) -> str:
    """
    Extract table name from schema document.
    First line starts with 'Table: <name>'.
    """
    if not doc:
        return ""
    first = doc.splitlines()[0].strip()
    m = TABLE_LINE_RE.search(first)
    return m.group(1).strip() if m else first


def _search_named_vector(
    vector_name: str,
    vector: List[float],
    limit: int,
):
    """Qdrant search using a named vector."""
    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/search"
    payload = {
        "vector": {
            "name": vector_name,
            "vector": vector,
        },
        "limit": limit,
        "with_payload": True,
    }

    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["result"]


def _format_hits(hits) -> List[Dict[str, Any]]:
    """Normalize Qdrant hits into stable output schema."""
    out = []
    for h in hits:
        payload = h.get("payload") or {}
        doc = payload.get(PAYLOAD_DOC_KEY, "") or ""
        out.append(
            {
                "table": extract_table_name(doc),
                "score": float(h["score"]),
                "doc": doc,
            }
        )
    return out


# Similarity search functions
def get_candidate_pref(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    E5 prefix mode (recommended):
      - Query embedding: "query: <query>"
      - Search vector:   "pref"
    """
    vec = model.encode(
        f"query: {query}",
        normalize_embeddings=False,
    ).tolist()

    hits = _search_named_vector(PREF_VECTOR_NAME, vec, top_k)
    return {
        "query": query,
        "candidates": _format_hits(hits),
    }


def get_candidate_nopref(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    E5 no-prefix baseline:
      - Query embedding: "<query>"
      - Search vector:   "nopref"
    """
    vec = model.encode(
        query,
        normalize_embeddings=False,
    ).tolist()

    hits = _search_named_vector(NOPREF_VECTOR_NAME, vec, top_k)
    return {
        "query": query,
        "candidates": _format_hits(hits),
    }

# Default similarity search function
def get_candidate_tables(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    return get_candidate_pref(query, top_k) # Using pref as it is perfoming better at K = 10


if __name__ == "__main__":
    q = input("Query: ").strip()

    print("\n=== Pref ===")
    out_pref = get_candidate_pref(q, top_k=TOP_K)
    print(json.dumps(out_pref, ensure_ascii=False, indent=2))

    print("\n=== No Pref ===")
    out_nopref = get_candidate_nopref(q, top_k=TOP_K)
    print(json.dumps(out_nopref, ensure_ascii=False, indent=2))