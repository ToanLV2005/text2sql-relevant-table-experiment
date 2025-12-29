"""
Evaluate jina-v4-embedder retrieval accuracy on 2 mode: single and multi vectors

This script:
  - Calls get_candidate_tables(query) that retrieve table using single vector
  - Calls get_candidate_multi_vector(query) that retrieve using multi vector
  - Use Recall@K
  - Reports per-query Recall@K and overall
"""

import os
import re
from typing import List, Tuple
from pipeline.get_candidate_jina import get_candidate_tables, get_candidate_multi_vector



# Config
TEST_SET_PATH = os.getenv("TEST_SET_PATH", "vi_test_set.txt")
OUTPUT_FILE = os.getenv("RETRIEVAL_REPORT_PATH","results/jina_retrieval_accuracy.txt")

K_VALUES = [1, 3, 5, 10]



# Helpers
def parse_test_set(path: str) -> List[Tuple[str, List[str]]]:
    """
    Query: <text>
    Tables: table_a, table_b
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = re.split(r"\n\s*\n", text.strip())
    pairs = []

    for b in blocks:
        q = re.search(r"Query:\s*(.+)", b)
        t = re.search(r"Tables:\s*(.+)", b)

        if not q or not t:
            continue

        query = q.group(1).strip()
        tables = [x.strip() for x in t.group(1).split(",") if x.strip()]
        pairs.append((query, tables))

    return pairs


def recall_at_k(retrieved: List[str], gold: List[str], k: int) -> float:
    """
    Recall@K = Number of retrieved table appear in required / Number of tables in required
    """
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(gold)


def write_line(path: str, line: str) -> None:
    """Append to output file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")



# Main evaluation
def main():
    # reset output file
    open(OUTPUT_FILE, "w", encoding="utf-8").close()

    pairs = parse_test_set(TEST_SET_PATH)
    n = len(pairs)

    write_line(OUTPUT_FILE, "=" * 80)
    write_line(OUTPUT_FILE, f"NUM QUERIES: {n}")
    write_line(OUTPUT_FILE, f"K_VALUES: {K_VALUES}")
    write_line(OUTPUT_FILE, "=" * 80)

    # Overall recall
    recall_single = {k: 0.0 for k in K_VALUES}
    recall_multi = {k: 0.0 for k in K_VALUES}

    for idx, (query, gold_tables) in enumerate(pairs, 1):
        gold_set = set(gold_tables)

        write_line(OUTPUT_FILE, "-" * 80)
        write_line(OUTPUT_FILE, f"[{idx}/{n}] QUERY: {query}")
        write_line(
            OUTPUT_FILE,
            f"GOLD TABLES ({len(gold_tables)}): {', '.join(gold_tables)}",
        )

        # SINGLE VECTOR
        single_res = get_candidate_tables(query)
        single_candidates = single_res.get("candidates", [])
        single_tables = [c["table"] for c in single_candidates]

        write_line(OUTPUT_FILE, "\n[SINGLE] Retrieved tables:")
        for i, c in enumerate(single_candidates, 1):
            mark = " IN" if c["table"] in gold_set else ""
            write_line(
                OUTPUT_FILE,
                f"  {i:>2}. {c['table']:<25} score={c['score']:.4f}{mark}",
            )

        for k in K_VALUES:
            r = recall_at_k(single_tables, gold_tables, k)
            recall_single[k] += r
            write_line(
                OUTPUT_FILE,
                f"[SINGLE] Recall@{k:<2}: {r:.3f}",
            )

        # MULTI VECTOR
        multi_res = get_candidate_multi_vector(query)
        multi_candidates = multi_res.get("candidates", [])
        multi_tables = [c["table"] for c in multi_candidates]

        write_line(OUTPUT_FILE, "\n[MULTI] Retrieved tables:")
        for i, c in enumerate(multi_candidates, 1):
            mark = " IN" if c["table"] in gold_set else ""
            write_line(
                OUTPUT_FILE,
                f"  {i:>2}. {c['table']:<25} score={c['score']:.4f}{mark}",
            )

        for k in K_VALUES:
            r = recall_at_k(multi_tables, gold_tables, k)
            recall_multi[k] += r
            write_line(
                OUTPUT_FILE,
                f"[MULTI]  Recall@{k:<2}: {r:.3f}",
            )

        write_line(OUTPUT_FILE, "")


    # Overall summary
    write_line(OUTPUT_FILE, "=" * 80)
    write_line(OUTPUT_FILE, "OVERALL RESULTS")
    write_line(OUTPUT_FILE, "=" * 80)

    write_line(OUTPUT_FILE, "\nSINGLE VECTOR:")
    for k in K_VALUES:
        write_line(
            OUTPUT_FILE,
            f"Recall@{k:<2}: {recall_single[k] / max(1, n):.4f}",
        )

    write_line(OUTPUT_FILE, "\nMULTI VECTOR:")
    for k in K_VALUES:
        write_line(
            OUTPUT_FILE,
            f"Recall@{k:<2}: {recall_multi[k] / max(1, n):.4f}",
        )



if __name__ == "__main__":
    main()
