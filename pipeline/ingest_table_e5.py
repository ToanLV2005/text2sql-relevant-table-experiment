"""
ingest_table.py

Purpose:
  - Ingest table schema documents
  - Embed each table schema using multilingual-e5-large
  - Store vectors in Qdrant for similarity search

Vector strategy:
  - pref   : "passage: <doc>"  (query-time prefix enabled)
  - nopref : "<doc>"           (no prefix)

"""

import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct



# Config 
MODEL_NAME = os.getenv("EMBED_MODEL","intfloat/multilingual-e5-large")

VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_FILE = os.getenv("SCHEMAT_FILE", os.path.join(BASE_DIR, "vi_schema.txt"))

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_E5","vi_schema_E5")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")



# Init model and client
model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(url=QDRANT_URL)


# Load and parse schema
with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Each table schema is separated by a fixed delimiter
docs = [d.strip() for d in raw_text.split("=== TABLE ===") if d.strip()]
print(f"Parsed table docs: {len(docs)}")



# Delete collection if already existed
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        # Prefix
        "pref": VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
        # No prefix 
        "nopref": VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    },
)


# Embed
points = []

for idx, doc in enumerate(docs):
    # Add "passage" prefix 
    vec_pref = model.encode(
        f"passage: {doc}",
        normalize_embeddings=False,
    ).tolist()

    # No prefix 
    vec_nopref = model.encode(
        doc,
        normalize_embeddings=False,
    ).tolist()

    points.append(
        PointStruct(
            id=idx,
            vector={
                "pref": vec_pref,
                "nopref": vec_nopref,
            },
            payload={
                # Store full schema text 
                "doc": doc,
            },
        )
    )

# Upsert to qdrant
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)
print(f"Upserted {len(points)} points into {COLLECTION_NAME}")