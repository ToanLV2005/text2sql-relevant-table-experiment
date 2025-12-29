"""
 E5-based table retrieval accuracy on 2 modes:
  - pref    : query-prefixed retrieval ("query: ...")
  - nopref  : raw query retrieval
"""

import os
import re
from typing import List, Tuple

from pipeline.get_candidate_e5 import (
    get_candidate_pref,
    get_candidate_nopref,
)



# Config
TEST_SET_PATH = os.getenv("TEST_SET_PATH", "vi_test_set.txt")
OUTPUT_FILE = os.getenv(
    "RETRIEVAL_REPORT_PATH_E5",
    "results/e5_retrieval_accuracy.txt",
)

K_VALUES = [1, 3, 5, 10]


# Helpers
def parse_test_set(path: str) -> List[Tuple[str, List[str]]]:
    """
    Expected format (blank-line separated):

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
    recall_pref = {k: 0.0 for k in K_VALUES}
    recall_nopref = {k: 0.0 for k in K_VALUES}

    for idx, (query, gold_tables) in enumerate(pairs, 1):
        gold_set = set(gold_tables)

        write_line(OUTPUT_FILE, "-" * 80)
        write_line(OUTPUT_FILE, f"[{idx}/{n}] QUERY: {query}")
        write_line(
            OUTPUT_FILE,
            f"GOLD TABLES ({len(gold_tables)}): {', '.join(gold_tables)}",
        )

        # PREF MODE
        pref_res = get_candidate_pref(query)
        pref_candidates = pref_res.get("candidates", [])
        pref_tables = [c["table"] for c in pref_candidates]

        write_line(OUTPUT_FILE, "\n[PREF] Retrieved tables:")
        for i, c in enumerate(pref_candidates, 1):
            mark = " IN" if c["table"] in gold_set else ""
            write_line(
                OUTPUT_FILE,
                f"  {i:>2}. {c['table']:<25} score={c['score']:.4f}{mark}",
            )

        for k in K_VALUES:
            r = recall_at_k(pref_tables, gold_tables, k)
            recall_pref[k] += r
            write_line(
                OUTPUT_FILE,
                f"[PREF] Recall@{k:<2}: {r:.3f}",
            )

        # NO-PREF MODE
        nopref_res = get_candidate_nopref(query)
        nopref_candidates = nopref_res.get("candidates", [])
        nopref_tables = [c["table"] for c in nopref_candidates]

        write_line(OUTPUT_FILE, "\n[NO-PREF] Retrieved tables:")
        for i, c in enumerate(nopref_candidates, 1):
            mark = " IN" if c["table"] in gold_set else ""
            write_line(
                OUTPUT_FILE,
                f"  {i:>2}. {c['table']:<25} score={c['score']:.4f}{mark}",
            )

        for k in K_VALUES:
            r = recall_at_k(nopref_tables, gold_tables, k)
            recall_nopref[k] += r
            write_line(
                OUTPUT_FILE,
                f"[NO-PREF] Recall@{k:<2}: {r:.3f}",
            )

        write_line(OUTPUT_FILE, "")

    # Overall summary
    write_line(OUTPUT_FILE, "=" * 80)
    write_line(OUTPUT_FILE, "OVERALL RESULTS (macro average Recall@K)")
    write_line(OUTPUT_FILE, "=" * 80)

    write_line(OUTPUT_FILE, "\nPREF MODE:")
    for k in K_VALUES:
        write_line(
            OUTPUT_FILE,
            f"Recall@{k:<2}: {recall_pref[k] / max(1, n):.4f}",
        )

    write_line(OUTPUT_FILE, "\nNO-PREF MODE:")
    for k in K_VALUES:
        write_line(
            OUTPUT_FILE,
            f"Recall@{k:<2}: {recall_nopref[k] / max(1, n):.4f}",
        )

    write_line(OUTPUT_FILE, "\nDONE.")
    write_line(OUTPUT_FILE, f"Report written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
