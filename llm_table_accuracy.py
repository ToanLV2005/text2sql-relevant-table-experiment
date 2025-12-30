"""
Evaluate table selection accuracy for the whole pipeline:
  - Retrieval candidates (vector search)
  - LLM final table selection
Metrics:
  - Recall
  - Precision
Outputs:
  - Per query breakdown
  - Overall averages
"""

import json
import re
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from dotenv import load_dotenv


from pipeline.get_final import get_final_tables
from pipeline.get_candidate_e5 import get_candidate_pref as get_candidate_tables

load_dotenv("pipeline/.env")
# Config
TEST_SET_PATH = "vi_test_set.txt"
LLM_TEST_FILE = os.getenv("LLM_TEST_OUT", "results/oss-120b-accuracy.txt")

TOP_K = int(os.getenv("TOP_K", "10"))  # shared with get_final.py
CANDIDATE_PRINT_TOP_N = 10 # how many to print in report

INCLUDE_REASONING = False # set True to dump LLM reasoning_content


@dataclass
class TestCase:
    query: str
    gold_tables: List[str]


# Helpers
def write_line(path: str, line: str) -> None:
    """Append to output file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_test_set(path: str) -> List[TestCase]:
    """
    Expected format:
        Query: <text>
        Tables: table1, table2
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    blocks = re.split(r"\n\s*\n", text)
    tests: List[TestCase] = []

    for b in blocks:
        q = re.search(r"Query:\s*(.+)", b)
        t = re.search(r"Tables:\s*(.+)", b)
        if not q or not t:
            continue

        query = q.group(1).strip()
        gold_tables = [x.strip() for x in t.group(1).split(",") if x.strip()]
        tests.append(TestCase(query=query, gold_tables=gold_tables))

    return tests


def recall(selected: List[str], gold: List[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(selected) & gold_set) / len(gold_set)


def precision(selected: List[str], gold: List[str]) -> float:
    if not selected:
        return 0.0
    gold_set = set(gold)
    return len(set(selected) & gold_set) / len(selected)



# Printing helpers
def print_candidates(
    out_file: str,
    candidates: List[dict],
    gold_tables: List[str],
    top_n: int,
) -> None:
    gold_set = set(gold_tables)

    """Output candidates from retrieval with score"""

    write_line(out_file, f"CANDIDATE TABLES (TOP-{top_n})")
    write_line(out_file, "-" * 60)

    for idx, c in enumerate(candidates[:top_n], 1):
        table = (c.get("table") or "").strip()
        score = c.get("score")
        mark = "IN" if table in gold_set else ""
        if score is None:
            write_line(out_file, f"{idx:02d}. {table:<35} {mark}")
        else:
            write_line(out_file, f"{idx:02d}. {table:<35} score={float(score):.4f} {mark}")

    write_line(out_file, "-" * 60)


def print_llm_table_check(
    out_file: str,
    llm_tables: List[str],
    gold_tables: List[str],
) -> None:
    """Output LLM result"""
    gold_set = set(gold_tables)
    write_line(out_file, "LLM TABLE CHECK")
    write_line(out_file, "-" * 60)

    for t in llm_tables:
        mark = "IN" if t in gold_set else ""
        write_line(out_file, f"- {t:<30} {mark}")

    write_line(out_file, "-" * 60)


# =========================
# Main
# =========================
def main() -> None:
    # Reset output file
    open(LLM_TEST_FILE, "w", encoding="utf-8").close()

    tests = parse_test_set(TEST_SET_PATH)
    if not tests:
        write_line(LLM_TEST_FILE, f"No test cases found in {TEST_SET_PATH}")
        return

    recalls: List[float] = []
    precisions: List[float] = []
    failures: List[Tuple[int, str]] = []

    # Token tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for i, tc in enumerate(tests, 1):
        query = tc.query
        gold_tables = tc.gold_tables

        write_line(LLM_TEST_FILE, "-" * 100)
        write_line(LLM_TEST_FILE, f"QUERY: {query}")
        write_line(
            LLM_TEST_FILE,
            f"REQUIRED TABLES ({len(gold_tables)}): {', '.join(gold_tables)}",
        )
        write_line(LLM_TEST_FILE, "")

        # Get candidates from retrieval
        try:
            cand = get_candidate_tables(query, top_k=TOP_K)
            candidates = cand.get("candidates", []) or []
        except Exception as e:
            candidates = []
            write_line(LLM_TEST_FILE, f"[ERROR] Retrieval failed: {e}")

        if candidates:
            print_candidates(
                out_file=LLM_TEST_FILE,
                candidates=candidates,
                gold_tables=gold_tables,
                top_n=min(CANDIDATE_PRINT_TOP_N, len(candidates)),
            )
        else:
            write_line(LLM_TEST_FILE, "CANDIDATE TABLES: (none)")

        write_line(LLM_TEST_FILE, "")

        # Get LLM final tables output
        try:
            result = get_final_tables(query)
            llm_tables = result.get("final_tables", []) or []
            reasoning = result.get("reasoning_content", "") or ""

            # Accumulate token usage
            token_usage = result.get("token_usage", {})
            total_prompt_tokens += token_usage.get("prompt_tokens", 0)
            total_completion_tokens += token_usage.get("completion_tokens", 0)
            total_tokens += token_usage.get("total_tokens", 0)
        except Exception as e:
            failures.append((i, str(e)))
            llm_tables = []
            reasoning = ""
            write_line(LLM_TEST_FILE, f"[ERROR] LLM failed: {e}")

        write_line(
            LLM_TEST_FILE,
            f"LLM TABLES ({len(llm_tables)}): {', '.join(llm_tables)}",
        )

        r = recall(llm_tables, gold_tables)
        p = precision(llm_tables, gold_tables)

        recalls.append(r)
        precisions.append(p)

        write_line(LLM_TEST_FILE, f"Recall:    {r:.3f}")
        write_line(LLM_TEST_FILE, f"Precision: {p:.3f}")
        write_line(LLM_TEST_FILE, "")

        print_llm_table_check(LLM_TEST_FILE, llm_tables, gold_tables)

        if INCLUDE_REASONING and reasoning:
            write_line(LLM_TEST_FILE, "")
            write_line(LLM_TEST_FILE, "LLM REASONING (raw)")
            write_line(LLM_TEST_FILE, "-" * 60)
            write_line(LLM_TEST_FILE, reasoning.strip())
            write_line(LLM_TEST_FILE, "-" * 60)

        write_line(LLM_TEST_FILE, "")

    # Summary
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    write_line(LLM_TEST_FILE, "=" * 100)
    write_line(LLM_TEST_FILE, "OVERALL AVERAGE METRICS (LLM)")
    write_line(LLM_TEST_FILE, "=" * 100)
    write_line(LLM_TEST_FILE, f"Test cases:          {len(tests)}")
    write_line(LLM_TEST_FILE, f"Average Recall:      {avg_recall:.4f}")
    write_line(LLM_TEST_FILE, f"Average Precision:   {avg_precision:.4f}")
    write_line(LLM_TEST_FILE, "")
    write_line(LLM_TEST_FILE, "TOKEN USAGE")
    write_line(LLM_TEST_FILE, f"Prompt tokens:       {total_prompt_tokens:,}")
    write_line(LLM_TEST_FILE, f"Completion tokens:   {total_completion_tokens:,}")
    write_line(LLM_TEST_FILE, f"Total tokens:        {total_tokens:,}")
    write_line(LLM_TEST_FILE, "=" * 100)

    if failures:
        write_line(LLM_TEST_FILE, "")
        write_line(LLM_TEST_FILE, "FAILURES")
        write_line(LLM_TEST_FILE, "-" * 100)
        for idx, msg in failures:
            write_line(LLM_TEST_FILE, f"- Case #{idx}: {msg}")


if __name__ == "__main__":
    main()
