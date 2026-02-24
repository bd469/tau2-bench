"""
Compute pass^k results table for all policies x agent models.

Usage:
    python experiments/compute_results_table.py
"""

import json
import math
import os
from pathlib import Path

EVAL_DIR = Path("evaluations/new_prompt")
BASE_DIR = Path("data/tau2/domains/retail")


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    if num_trials < k:
        return None
    return math.comb(success_count, k) / math.comb(num_trials, k)


def get_reward(sim: dict) -> float:
    """Extract reward from either file format."""
    # evaluate_prompt.py format
    if "reward" in sim and sim["reward"] is not None:
        return float(sim["reward"])
    # tau2 run format
    if "reward_info" in sim and sim["reward_info"]:
        return float(sim["reward_info"].get("reward", 0.0) or 0.0)
    return 0.0


def load_task_results(paths: list[Path]) -> dict[int, list[int]] | None:
    """
    Load and combine results from one or more eval files.
    Returns dict of task_id -> list of success (0/1) per trial, or None if no files found.
    """
    found = [p for p in paths if p.exists()]
    if not found:
        return None

    task_results: dict[int, list[int]] = {}

    for path in found:
        with open(path) as f:
            d = json.load(f)

        # evaluate_prompt.py format: has "task_results" key
        if "task_results" in d:
            for r in d["task_results"]:
                tid = r["task_id"]
                success = 1 if r.get("reward", 0.0) >= 0.999 else 0
                if tid not in task_results:
                    task_results[tid] = []
                task_results[tid].append(success)

        # tau2 run format: has "simulations" key
        elif "simulations" in d:
            for s in d["simulations"]:
                tid = s["task_id"]
                reward = get_reward(s)
                success = 1 if reward >= 0.999 else 0
                if tid not in task_results:
                    task_results[tid] = []
                task_results[tid].append(success)

    return task_results if task_results else None


def compute_pass_k(task_results: dict[int, list[int]]) -> dict[int, float]:
    """Compute pass^1, pass^2, pass^3 from task results."""
    n_tasks = len(task_results)
    results = {}
    for k in [1, 2, 3]:
        eligible = [v for v in task_results.values() if len(v) >= k]
        if not eligible:
            results[k] = None
        else:
            total = sum(pass_hat_k(len(v), sum(v), k) for v in eligible)
            results[k] = total / n_tasks * 100
    return results


def fmt(val) -> str:
    if val is None:
        return "  —  "
    return f"{val:.1f}"


# ── Define all rows ──────────────────────────────────────────────────────────

ROWS = [
    {
        "label": "Tudor + gpt-4o",
        "gpt4o_files": [
            EVAL_DIR / "eval_tudor_gpt4o-refined_gpt-4o_test_1trial.json",
            EVAL_DIR / "eval_tudor_gpt4o-refined_gpt-4o_test_2trial.json",
            EVAL_DIR / "eval_tudor_gpt4o-refined_gpt-4o_test_3trial.json",
        ],
        "gpt5mini_files": [
            EVAL_DIR / "eval_tudor_gpt4o-refined_gpt-5-mini_test_3trials.json",
        ],
    },
    {
        "label": "Chao + gpt-4o",
        "gpt4o_files": [
            EVAL_DIR / "eval_chao_gpt4o-refined_gpt-4o_test_3trials.json",
        ],
        "gpt5mini_files": [
            EVAL_DIR / "eval_chao_gpt4o-refined_gpt-5-mini_test_3trials.json",
        ],
    },
    {
        "label": "Chloe + gpt-4o",
        "gpt4o_files": [
            EVAL_DIR / "eval_chloe_gpt4o-refined_gpt-4o_test_1trial.json",
            EVAL_DIR / "eval_chloe_gpt4o-refined_gpt-4o_test_2trial.json",
            EVAL_DIR / "eval_chloe_gpt4o-refined_gpt-4o_test_3trial.json",
        ],
        "gpt5mini_files": [
            EVAL_DIR / "eval_chloe_gpt4o-refined_gpt-5-mini_test_3trials.json",
        ],
    },
    {
        "label": "Tudor + gpt-5-mini",
        "gpt4o_files": [
            EVAL_DIR / "eval_tudor_gpt5mini-refined_gpt-4o_test_1trial.json",
            EVAL_DIR / "eval_tudor_gpt5mini-refined_gpt-4o_test_2trial.json",
            EVAL_DIR / "eval_tudor_gpt5mini-refined_gpt-4o_test_3trial.json",
        ],
        "gpt5mini_files": [
            EVAL_DIR / "eval_tudor_gpt5mini-refined_gpt-5-mini_test_3trials.json",
        ],
    },
    {
        "label": "Chao + gpt-5-mini",
        "gpt4o_files": [
            EVAL_DIR / "eval_chao_gpt5mini-refined_gpt-4o_test_3trials.json",
        ],
        "gpt5mini_files": [
            EVAL_DIR / "eval_chao_gpt5mini-refined_gpt-5-mini_test_3trials.json",
        ],
    },
    {
        "label": "Chloe + gpt-5-mini",
        "gpt4o_files": [
            EVAL_DIR / "eval_chloe_gpt5mini-refined_gpt-4o_test_1trial.json",
            EVAL_DIR / "eval_chloe_gpt5mini-refined_gpt-4o_test_2trial.json",
            EVAL_DIR / "eval_chloe_gpt5mini-refined_gpt-4o_test_3trial.json",
        ],
        "gpt5mini_files": [
            EVAL_DIR / "eval_chloe_gpt5mini-refined_gpt-5-mini_test_3trials.json",
        ],
    },
    {
        "label": "Base policy",
        "gpt4o_files": [
            EVAL_DIR / "eval_base-policy_gpt-4o_test_3trials.json",
        ],
        "gpt5mini_files": [
            EVAL_DIR / "eval_base-policy_gpt-5-mini_test_3trials.json",
        ],
    },
]


def print_table(agent_model: str, file_key: str):
    col_w = [20, 10, 10, 10]
    header = f"{'Policy':<{col_w[0]}}{'pass^1':>{col_w[1]}}{'pass^2':>{col_w[2]}}{'pass^3':>{col_w[3]}}"
    sep = "-" * sum(col_w)

    print(f"\nAgent model = {agent_model}")
    print(sep)
    print(header)
    print(sep)

    for row in ROWS:
        task_results = load_task_results(row[file_key])
        if task_results is None:
            p1, p2, p3 = "  —  ", "  —  ", "  —  "
        else:
            pk = compute_pass_k(task_results)
            p1, p2, p3 = fmt(pk[1]), fmt(pk[2]), fmt(pk[3])
        print(f"{row['label']:<{col_w[0]}}{p1:>{col_w[1]}}{p2:>{col_w[2]}}{p3:>{col_w[3]}}")

    print(sep)


if __name__ == "__main__":
    print_table("gpt-4o", "gpt4o_files")
    print_table("gpt-5-mini", "gpt5mini_files")
