"""
Purpose:
  - Read schema docs from a text file (blocks separated by "=== TABLE ===")
  - Embed each block using jina-embeddings-v4
  - Store embeddings in Qdrant under 2 vector names:
      1) "single": single-vector
      2) "multi" : multi-vector 
"""

import os
import torch
from transformers import AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, MultiVectorConfig, MultiVectorComparator

from dotenv import load_dotenv

# Config from env
load_dotenv()
BASE_DIR = os.path.dirname(__file__)
MODEL_NAME = os.getenv("JINA_MODEL", "jinaai/jina-embeddings-v4")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION_JINA", "vi_schema_jina_v4")
SCHEMA_FILE = os.getenv("SCHEMA_FILE", os.path.join(BASE_DIR, "vi_schema.txt"))

# Vector dimension
SINGLE_DIM = int(os.getenv("SINGLE_DIM", "1024"))

# Embedding max length
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

# Device init
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# trust_remote_code=True is required for jina-embeddings-v4 encode_text()
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

# Qdrant client
client = QdrantClient(url=QDRANT_URL)


# Load docs
# Expected format: blocks separated by "=== TABLE ==="
with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    raw = f.read()

docs = [d.strip() for d in raw.split("=== TABLE ===") if d.strip()]
print("Parsed table docs:", len(docs))


# Delete if collection already exist
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

# 2 vector namespace for single and multi
client.create_collection(
    collection_name=COLLECTION,
    vectors_config={
        "single": VectorParams(size=SINGLE_DIM, distance=Distance.COSINE),
        "multi": VectorParams(
            size=128,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    },
)


# jina-embedding-v4 returns embedded text as Pytorch Tensors
# Qdrant can't store Pytorch Tensors
# Need helpers to convert to python list
def to_list_1d(x: torch.Tensor):
    """Convert (dim,) tensor -> Python list[float]."""
    return x.detach().float().cpu().numpy().tolist()


def to_list_2d(x: torch.Tensor):
    """Convert (n_tokens, 128) tensor -> Python list[list[float]]."""
    return x.detach().float().cpu().numpy().tolist()


# Embed points into single and multi vector namespaces
points = []
for idx, doc in enumerate(docs):
    # Produces a single vector for each doc
    single_vec = model.encode_text(
        doc,
        task="retrieval",
        truncate_dim=SINGLE_DIM,
        max_length=MAX_LENGTH,
    )

    # ---- multi-vector embedding ----
    # Produces multi vectors for each doc (n_tokens, 128)
    multi_vec = model.encode_text(
        doc,
        task="retrieval",
        return_multivector=True,
        max_length=MAX_LENGTH,
    )

    # Store both under named vectors
    # payload keeps original doc text
    points.append(
        PointStruct(
            id=idx,
            vector={
                "single": to_list_1d(single_vec),
                "multi": to_list_2d(multi_vec),
            },
            payload={"doc": doc},
        )
    )


# Upsert all points into qdrant
client.upsert(collection_name=COLLECTION, points=points)
print(f"Upserted {len(points)} points into {COLLECTION}")
