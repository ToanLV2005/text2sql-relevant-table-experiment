"""
2 modes:
    - Single vector
    - Multi vector
"""

import os
import re
import json
from typing import List, Dict, Any

import torch
from transformers import AutoModel
from qdrant_client import QdrantClient
from dotenv import load_dotenv



# Config from env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "vi_schema_jina_v4")
EMBEDDER_NAME = os.getenv("EMBEDDER_NAME", "jinaai/jina-embeddings-v4")

SINGLE_DIM = int(os.getenv("SINGLE_DIM", "1024"))
TOP_K = int(os.getenv("TOP_K", "10"))

# Must match vector names created during ingest
SINGLE_USING = os.getenv("SINGLE_USING", "single")
MULTI_USING = os.getenv("MULTI_USING", "multi")

# Payload key where we stored full schema doc
PAYLOAD_DOC_KEY = os.getenv("PAYLOAD_DOC_KEY", "doc")


# Init clients + model
client = QdrantClient(url=QDRANT_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    EMBEDDER_NAME,
    trust_remote_code=True,  # required for jina encode_text()
).to(device)

# Regex to match first line of each schema doc
TABLE_LINE_RE = re.compile(r"^\s*Table:\s*(.+?)\s*$", re.IGNORECASE)


# Helpers
def extract_table_name(payload_doc: str) -> str:
    """
    Full schema doc is stored in payload["doc"].
    Convention: first line contains 'Table: <name>'.
    """
    if not payload_doc:
        return ""
    first = payload_doc.splitlines()[0].strip()
    m = TABLE_LINE_RE.search(first)
    return m.group(1).strip() if m else first


# Embedding
def embed_single(text: str) -> List[float]:
    """
    Embed query into a single dense vector.
    Returns list[float] for Qdrant.
    """
    out = model.encode_text(
        text,
        task="retrieval",
        truncate_dim=SINGLE_DIM,
        max_length=512,
    )

    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Unexpected single output type: {type(out)}")

    return out.detach().float().cpu().numpy().tolist()


def embed_multi(text: str) -> List[List[float]]:
    """
    Embed query into multi-vector (late interaction).
    Returns list[list[float]] for Qdrant.
    """
    out = model.encode_text(
        text,
        task="retrieval",
        return_multivector=True,
        max_length=512,
    )

    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Unexpected multi output type: {type(out)}")

    vec = out.detach().float().cpu().numpy().tolist()

    if not (isinstance(vec, list) and vec and isinstance(vec[0], list)):
        raise RuntimeError("Multi-vector embedding must be 2D list")

    return vec


# Qdrant queries
def query_single(vec_single: List[float], limit: int):
    """Query Qdrant using single vector."""
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec_single,
        using=SINGLE_USING,
        limit=limit,
        with_payload=True,
    )
    return res.points


def query_multi(vec_multi: List[List[float]], limit: int):
    """Query Qdrant using multi-vector"""
    res = client.query_points(
        collection_name=COLLECTION,
        query=vec_multi,
        using=MULTI_USING,
        limit=limit,
        with_payload=True,
    )
    return res.points


def _format_hits(pts) -> List[Dict[str, Any]]:
    """Format hits"""
    hits = []
    for p in pts:
        payload = p.payload or {}
        doc = payload.get(PAYLOAD_DOC_KEY, "") or ""
        hits.append(
            {
                "table": extract_table_name(doc),
                "score": float(p.score),
                "doc": doc,
            }
        )
    return hits


def get_candidate_tables_single_vector(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Single vector retrieval
    Returns:
      {
        "query": str,
        "candidates": [{"table": str, "score": float, "doc": str}, ...]
      }
    """
    vec = embed_single(query)
    pts = query_single(vec, limit=top_k)
    return {"query": query, "candidates": _format_hits(pts)}


def get_candidate_multi_vector(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Multi vector retrieval 
    """
    vec = embed_multi(query)
    pts = query_multi(vec, limit=top_k)
    return {"query": query, "candidates": _format_hits(pts)}

def get_candidate_tables(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    return get_candidate_tables_single_vector(query, top_k) # Default single as it is performing better




if __name__ == "__main__":
    q = input("Query: ").strip()
    print("\n=== SINGLE VECTOR ===")
    out_single = get_candidate_tables(q, top_k=TOP_K)
    print(json.dumps(out_single, ensure_ascii=False, indent=2))

    print("\n=== MULTI VECTOR ===")
    out_multi = get_candidate_multi_vector(q, top_k=TOP_K)
    print(json.dumps(out_multi, ensure_ascii=False, indent=2))
