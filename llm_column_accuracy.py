"""
Evaluate column selection accuracy for the pipeline:
  - Table selection (from previous step)
  - LLM column selection

Metrics:
  - Recall: Are all required columns selected?
  - Precision: Are selected columns relevant?

Outputs:
  - Per query breakdown
  - Overall averages
"""

import json
import re
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


from dotenv import load_dotenv

from pipeline.get_columns import get_final_columns, get_columns_from_tables

load_dotenv("pipeline/.env")

# Config
TEST_SET_PATH = os.getenv("COLUMN_TEST_SET", "vi_test_set.txt")
OUTPUT_FILE = os.getenv("COLUMN_TEST_OUT", "results/en_column_accuracy.txt")
TOP_K = int(os.getenv("TOP_K", "10"))
INCLUDE_REASONING = True


@dataclass
class ColumnTestCase:
    query: str
    gold_tables: List[str]
    gold_columns: Dict[str, List[str]]  # {table_name: [col1, col2, ...]}


def write_line(path: str, line: str) -> None:
    """Append to output file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_column_test_set(path: str) -> List[ColumnTestCase]:
    """
    Expected format:
        Query: <text>
        Tables: table1, table2
        Columns:
        - table1: col1, col2, col3
        - table2: colA, colB
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    blocks = re.split(r"\n\s*\n", text)
    tests: List[ColumnTestCase] = []

    for b in blocks:
        # Parse query
        q = re.search(r"Query:\s*(.+)", b)
        if not q:
            continue

        query = q.group(1).strip()

        # Parse tables
        t = re.search(r"Tables:\s*(.+)", b)
        gold_tables = []
        if t:
            gold_tables = [x.strip() for x in t.group(1).split(",") if x.strip()]

        # Parse columns
        gold_columns: Dict[str, List[str]] = {}
        col_section = re.search(r"Columns:\s*\n((?:- .+\n?)+)", b)
        if col_section:
            col_lines = col_section.group(1).strip().split("\n")
            for line in col_lines:
                # Format: - table_name: col1, col2, col3
                match = re.match(r"-\s*(\w+)\s*:\s*(.+)", line)
                if match:
                    table_name = match.group(1).strip()
                    cols = [c.strip() for c in match.group(2).split(",") if c.strip()]
                    gold_columns[table_name] = cols

        tests.append(ColumnTestCase(
            query=query,
            gold_tables=gold_tables,
            gold_columns=gold_columns
        ))

    return tests


def column_recall(selected: Dict[str, List[str]], gold: Dict[str, List[str]]) -> float:
    """
    Recall: proportion of gold columns that were selected.
    """
    total_gold = 0
    matched = 0

    for table, cols in gold.items():
        total_gold += len(cols)
        if table in selected:
            matched += len(set(selected[table]) & set(cols))

    return matched / total_gold if total_gold > 0 else 0.0


def column_precision(selected: Dict[str, List[str]], gold: Dict[str, List[str]]) -> float:
    """
    Precision: proportion of selected columns that are in gold.
    """
    total_selected = sum(len(cols) for cols in selected.values())
    if total_selected == 0:
        return 0.0

    matched = 0
    for table, cols in selected.items():
        if table in gold:
            matched += len(set(cols) & set(gold[table]))

    return matched / total_selected


def print_column_comparison(
    out_file: str,
    selected: Dict[str, List[str]],
    gold: Dict[str, List[str]],
) -> None:
    """Output column selection comparison."""
    write_line(out_file, "COLUMN COMPARISON")
    write_line(out_file, "-" * 60)

    all_tables = set(selected.keys()) | set(gold.keys())

    for table in sorted(all_tables):
        sel_cols = set(selected.get(table, []))
        gold_cols = set(gold.get(table, []))

        write_line(out_file, f"TABLE: {table}")

        # Columns in both (correct)
        correct = sel_cols & gold_cols
        # Columns in gold but not selected (missed)
        missed = gold_cols - sel_cols
        # Columns selected but not in gold (extra)
        extra = sel_cols - gold_cols

        if correct:
            write_line(out_file, f"  [OK]     {', '.join(sorted(correct))}")
        if missed:
            write_line(out_file, f"  [MISSED] {', '.join(sorted(missed))}")
        if extra:
            write_line(out_file, f"  [EXTRA]  {', '.join(sorted(extra))}")

        write_line(out_file, "")

    write_line(out_file, "-" * 60)


def main() -> None:
    # Reset output file
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    open(OUTPUT_FILE, "w", encoding="utf-8").close()

    if not os.path.exists(TEST_SET_PATH):
        write_line(OUTPUT_FILE, f"Test set not found: {TEST_SET_PATH}")
        write_line(OUTPUT_FILE, "")
        write_line(OUTPUT_FILE, "Please create a test set file with format:")
        write_line(OUTPUT_FILE, "")
        write_line(OUTPUT_FILE, "Query: <question>")
        write_line(OUTPUT_FILE, "Tables: table1, table2")
        write_line(OUTPUT_FILE, "Columns:")
        write_line(OUTPUT_FILE, "- table1: col1, col2, col3")
        write_line(OUTPUT_FILE, "- table2: colA, colB")
        print(f"Test set not found: {TEST_SET_PATH}")
        return

    tests = parse_column_test_set(TEST_SET_PATH)
    if not tests:
        write_line(OUTPUT_FILE, f"No test cases found in {TEST_SET_PATH}")
        return

    recalls: List[float] = []
    precisions: List[float] = []
    failures: List[Tuple[int, str]] = []

    # Micro-average tracking (total counts across all queries)
    total_gold_columns = 0
    total_selected_columns = 0
    total_matched_columns = 0

    # Token tracking
    total_table_prompt = 0
    total_table_completion = 0
    total_col_prompt = 0
    total_col_completion = 0

    for i, tc in enumerate(tests, 1):
        write_line(OUTPUT_FILE, "=" * 100)
        write_line(OUTPUT_FILE, f"TEST CASE #{i}")
        write_line(OUTPUT_FILE, "=" * 100)
        write_line(OUTPUT_FILE, f"QUERY: {tc.query}")
        write_line(OUTPUT_FILE, f"REQUIRED TABLES: {', '.join(tc.gold_tables)}")
        write_line(OUTPUT_FILE, "")
        write_line(OUTPUT_FILE, "REQUIRED COLUMNS:")
        for table, cols in tc.gold_columns.items():
            write_line(OUTPUT_FILE, f"  {table}: {', '.join(cols)}")
        write_line(OUTPUT_FILE, "")

        try:
            # Run full pipeline (table + column selection)
            result = get_final_columns(tc.query, top_k=TOP_K)

            selected_tables = result.get("final_tables", [])
            selected_columns = result.get("columns", {})

            # Token usage
            token_usage = result.get("token_usage", {})
            table_tokens = token_usage.get("table_selection", {})
            col_tokens = token_usage.get("column_selection", {})

            total_table_prompt += table_tokens.get("prompt_tokens", 0)
            total_table_completion += table_tokens.get("completion_tokens", 0)
            total_col_prompt += col_tokens.get("prompt_tokens", 0)
            total_col_completion += col_tokens.get("completion_tokens", 0)

        except Exception as e:
            failures.append((i, str(e)))
            selected_tables = []
            selected_columns = {}
            write_line(OUTPUT_FILE, f"[ERROR] Pipeline failed: {e}")

        write_line(OUTPUT_FILE, f"SELECTED TABLES: {', '.join(selected_tables)}")
        write_line(OUTPUT_FILE, "")
        write_line(OUTPUT_FILE, "SELECTED COLUMNS:")
        for table, cols in selected_columns.items():
            write_line(OUTPUT_FILE, f"  {table}: {', '.join(cols)}")
        write_line(OUTPUT_FILE, "")

        # Calculate metrics
        r = column_recall(selected_columns, tc.gold_columns)
        p = column_precision(selected_columns, tc.gold_columns)

        recalls.append(r)
        precisions.append(p)

        # Accumulate counts for micro-average
        for table, cols in tc.gold_columns.items():
            total_gold_columns += len(cols)
            if table in selected_columns:
                total_matched_columns += len(set(selected_columns[table]) & set(cols))
        for table, cols in selected_columns.items():
            total_selected_columns += len(cols)

        write_line(OUTPUT_FILE, f"Column Recall:    {r:.3f}")
        write_line(OUTPUT_FILE, f"Column Precision: {p:.3f}")
        write_line(OUTPUT_FILE, "")

        print_column_comparison(OUTPUT_FILE, selected_columns, tc.gold_columns)

        if INCLUDE_REASONING:
            reasoning = result.get("column_selection_reasoning", "")
            if reasoning:
                write_line(OUTPUT_FILE, "")
                write_line(OUTPUT_FILE, "LLM REASONING")
                write_line(OUTPUT_FILE, "-" * 60)
                write_line(OUTPUT_FILE, reasoning.strip())
                write_line(OUTPUT_FILE, "-" * 60)

        write_line(OUTPUT_FILE, "")

    # Summary
    # Macro-average (average of per-query metrics)
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0

    # Micro-average (total matched / total gold or selected)
    avg_recall = total_matched_columns / total_gold_columns if total_gold_columns > 0 else 0.0
    avg_precision = total_matched_columns / total_selected_columns if total_selected_columns > 0 else 0.0

    write_line(OUTPUT_FILE, "=" * 100)
    write_line(OUTPUT_FILE, "OVERALL METRICS (COLUMN SELECTION)")
    write_line(OUTPUT_FILE, "=" * 100)
    write_line(OUTPUT_FILE, f"Test cases:              {len(tests)}")
    write_line(OUTPUT_FILE, "")
    write_line(OUTPUT_FILE, "MICRO-AVERAGE (weighted by column count):")
    write_line(OUTPUT_FILE, f"  Total gold columns:    {total_gold_columns}")
    write_line(OUTPUT_FILE, f"  Total selected columns:{total_selected_columns}")
    write_line(OUTPUT_FILE, f"  Total matched columns: {total_matched_columns}")
    write_line(OUTPUT_FILE, f"  Recall:                {avg_recall:.4f}")
    write_line(OUTPUT_FILE, f"  Precision:             {avg_precision:.4f}")
    write_line(OUTPUT_FILE, "")
    write_line(OUTPUT_FILE, "MACRO-AVERAGE (average per query):")
    write_line(OUTPUT_FILE, f"  Recall:                {macro_recall:.4f}")
    write_line(OUTPUT_FILE, f"  Precision:             {macro_precision:.4f}")
    write_line(OUTPUT_FILE, "")
    write_line(OUTPUT_FILE, "TOKEN USAGE")
    write_line(OUTPUT_FILE, f"Table Selection:")
    write_line(OUTPUT_FILE, f"  Prompt tokens:         {total_table_prompt:,}")
    write_line(OUTPUT_FILE, f"  Completion tokens:     {total_table_completion:,}")
    write_line(OUTPUT_FILE, f"Column Selection:")
    write_line(OUTPUT_FILE, f"  Prompt tokens:         {total_col_prompt:,}")
    write_line(OUTPUT_FILE, f"  Completion tokens:     {total_col_completion:,}")
    write_line(OUTPUT_FILE, f"Total tokens:            {total_table_prompt + total_table_completion + total_col_prompt + total_col_completion:,}")
    write_line(OUTPUT_FILE, "=" * 100)

    if failures:
        write_line(OUTPUT_FILE, "")
        write_line(OUTPUT_FILE, "FAILURES")
        write_line(OUTPUT_FILE, "-" * 100)
        for idx, msg in failures:
            write_line(OUTPUT_FILE, f"- Case #{idx}: {msg}")

    print(f"Results written to: {OUTPUT_FILE}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")


if __name__ == "__main__":
    main()
