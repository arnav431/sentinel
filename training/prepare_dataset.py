#!/usr/bin/env python3
"""Build train/val/test JSONL files from the seed examples.

Usage:
    python training/prepare_dataset.py \\
        --seed data/training/seed_examples.py \\
        --out-dir data/training/splits \\
        --val-fraction 0.15 --test-fraction 0.15 --seed-value 1337

Splitting is done by `scenario_id`, never by individual example, so that
near-duplicate states from the same fictional lab machine can never
appear on both sides of a split (the leakage `validate_dataset.py` checks
for). This matters more than it sounds: within one scenario, consecutive
examples share almost all of their `context` by construction (same
target, overlapping service lists), so splitting at the example level
would let the model "learn the scenario" rather than generalize.

Each output record is a single training example in a standard
instruction-tuning chat shape:
    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                   {"role": "assistant", "content": "<json-encoded StructuredDecision>"}],
     "meta": {"scenario_id": ..., "category": ..., "synthetic": true}}

`train_lora.py` consumes this format directly.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sentinel.schemas import StructuredDecision  # noqa: E402

SYSTEM_PROMPT = (
    "You are Sentinel, an assistant that helps a student investigate an "
    "explicitly authorized beginner-level lab machine (e.g. Hack The Box). "
    "You never fabricate tool output, credentials, flags, or shell access. "
    "You only ever propose actions; you cannot execute anything yourself. "
    "Respond with ONLY a single JSON object matching the StructuredDecision "
    "schema — no prose, no markdown fences."
)


def _load_seed_module(path: Path):
    spec = importlib.util.spec_from_file_location("seed_examples", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def build_record(example: dict[str, Any]) -> dict[str, Any]:
    if not example.get("synthetic"):
        raise ValueError(
            f"Example for scenario '{example.get('scenario_id')}' does not declare "
            f"synthetic=True. Refusing to emit — every example must be explicitly "
            f"labeled so a reader can never mistake it for real tool output."
        )

    # Validate the label against the real Pydantic schema the inference
    # path uses. A malformed label would otherwise silently teach the
    # model an invalid output shape.
    decision = StructuredDecision.model_validate(example["target_decision"])

    user_content = (
        "Assessment state (JSON):\n" + json.dumps(example["context"], indent=2)
    )
    assistant_content = decision.model_dump_json()

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta": {
            "scenario_id": example["scenario_id"],
            "category": example.get("category", "uncategorized"),
            "synthetic": True,
            "has_finding_label": "finding_label" in example,
        },
    }


def split_by_scenario(
    examples: list[dict[str, Any]],
    val_fraction: float,
    test_fraction: float,
    seed_value: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    scenario_ids = sorted({ex["scenario_id"] for ex in examples})
    rng = random.Random(seed_value)
    rng.shuffle(scenario_ids)

    n = len(scenario_ids)
    n_test = max(1, round(n * test_fraction)) if n > 2 else 0
    n_val = max(1, round(n * val_fraction)) if n > 2 else 0
    # Keep at least one scenario in train when the pool is tiny.
    n_test = min(n_test, max(0, n - 2))
    n_val = min(n_val, max(0, n - n_test - 1))

    test_ids = set(scenario_ids[:n_test])
    val_ids = set(scenario_ids[n_test:n_test + n_val])
    train_ids = set(scenario_ids) - test_ids - val_ids

    train = [ex for ex in examples if ex["scenario_id"] in train_ids]
    val = [ex for ex in examples if ex["scenario_id"] in val_ids]
    test = [ex for ex in examples if ex["scenario_id"] in test_ids]
    return train, val, test


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=REPO_ROOT / "data/training/seed_examples.py")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/training/splits")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed-value", type=int, default=1337, help="Fixed random seed for reproducible splits.")
    args = parser.parse_args()

    module = _load_seed_module(args.seed)
    examples: list[dict[str, Any]] = module.EXAMPLES
    if not examples:
        print("No examples found in seed file.", file=sys.stderr)
        return 1

    train, val, test = split_by_scenario(examples, args.val_fraction, args.test_fraction, args.seed_value)

    train_scenarios = {ex["scenario_id"] for ex in train}
    val_scenarios = {ex["scenario_id"] for ex in val}
    test_scenarios = {ex["scenario_id"] for ex in test}
    overlap = (train_scenarios & val_scenarios) | (train_scenarios & test_scenarios) | (val_scenarios & test_scenarios)
    if overlap:
        print(f"FATAL: scenario leakage across splits: {overlap}", file=sys.stderr)
        return 1

    for name, split in (("train", train), ("val", val), ("test", test)):
        records = [build_record(ex) for ex in split]
        write_jsonl(records, args.out_dir / f"{name}.jsonl")
        print(f"{name}: {len(records)} examples across {len({r['meta']['scenario_id'] for r in records})} scenarios "
              f"-> {args.out_dir / f'{name}.jsonl'}")

    print(f"\nTotal: {len(examples)} examples, {len({ex['scenario_id'] for ex in examples})} scenarios. "
          f"Split seed: {args.seed_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
