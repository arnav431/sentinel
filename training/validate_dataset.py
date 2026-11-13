#!/usr/bin/env python3
"""Validate prepared JSONL dataset files before they're used for training.

Checks performed on each split file and across all of them together:
  1. Every record parses as JSON and matches the expected chat-message shape.
  2. The assistant message parses as a valid `StructuredDecision`.
  3. Every `proposed_action.tool` names a real "tool_name.operation" pair
     registered in the current tool registry (a label that references a
     tool/operation that doesn't exist would teach the model to propose
     something the policy engine can only ever reject).
  4. Every record is marked synthetic — nothing here should ever be
     mistaken for genuine captured tool output.
  5. No scenario_id appears in more than one split file (leakage).
  6. No record contains an obvious secret (API key / private key / AWS
     key / password= pattern) — defense in depth even though the seed
     examples are synthetic and shouldn't contain any.

Exits non-zero and prints every failure found (not just the first) so a
dataset author gets the full picture in one pass.

Usage:
    python training/validate_dataset.py --splits-dir data/training/splits
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sentinel.schemas import StructuredDecision  # noqa: E402
from sentinel.tools.base import default_registry  # noqa: E402

SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----"),
    re.compile(r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]{4,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
]


class ValidationIssue:
    def __init__(self, split: str, line_no: int, message: str):
        self.split = split
        self.line_no = line_no
        self.message = message

    def __str__(self) -> str:
        return f"[{self.split}:{self.line_no}] {self.message}"


def _registered_pairs() -> set[str]:
    registry = default_registry()
    pairs = set()
    for tool in registry.all():
        for op_name in tool.operations():
            pairs.add(f"{tool.name}.{op_name}")
    return pairs


def validate_file(path: Path, registered: set[str]) -> tuple[list[ValidationIssue], set[str]]:
    issues: list[ValidationIssue] = []
    scenario_ids: set[str] = set()

    if not path.exists():
        return [ValidationIssue(path.name, 0, "file does not exist")], scenario_ids

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(path.name, line_no, f"invalid JSON: {exc}"))
            continue

        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            issues.append(ValidationIssue(path.name, line_no, "expected exactly 3 messages (system/user/assistant)"))
            continue
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            issues.append(ValidationIssue(path.name, line_no, f"unexpected roles order: {roles}"))

        meta = record.get("meta", {})
        if not meta.get("synthetic"):
            issues.append(ValidationIssue(path.name, line_no, "record not marked synthetic=true"))
        scenario_id = meta.get("scenario_id")
        if not scenario_id:
            issues.append(ValidationIssue(path.name, line_no, "missing meta.scenario_id"))
        else:
            scenario_ids.add(scenario_id)

        assistant_content = messages[2].get("content", "") if len(messages) == 3 else ""
        try:
            decision_data = json.loads(assistant_content)
            decision = StructuredDecision.model_validate(decision_data)
        except Exception as exc:
            issues.append(ValidationIssue(path.name, line_no, f"assistant content is not a valid StructuredDecision: {exc}"))
            decision = None

        if decision is not None and decision.proposed_action is not None:
            tool_key = decision.proposed_action.tool
            if tool_key not in registered:
                issues.append(ValidationIssue(
                    path.name, line_no,
                    f"proposed_action.tool '{tool_key}' is not a registered tool.operation pair "
                    f"(registered: {sorted(registered)})",
                ))

        raw_text = line
        for pattern in SECRET_PATTERNS:
            if pattern.search(raw_text):
                issues.append(ValidationIssue(path.name, line_no, f"possible secret matched pattern {pattern.pattern!r}"))

    return issues, scenario_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "data/training/splits")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    registered = _registered_pairs()
    all_issues: list[ValidationIssue] = []
    scenario_ids_by_split: dict[str, set[str]] = {}

    for split in args.splits:
        path = args.splits_dir / f"{split}.jsonl"
        issues, scenario_ids = validate_file(path, registered)
        all_issues.extend(issues)
        scenario_ids_by_split[split] = scenario_ids
        status = "OK" if not issues else f"{len(issues)} issue(s)"
        print(f"{split}: {status} ({len(scenario_ids)} scenario(s))")

    # Cross-split leakage check.
    splits = list(scenario_ids_by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = scenario_ids_by_split[splits[i]] & scenario_ids_by_split[splits[j]]
            if overlap:
                all_issues.append(ValidationIssue(
                    "cross-split", 0,
                    f"scenario_id leakage between '{splits[i]}' and '{splits[j]}': {sorted(overlap)}",
                ))

    if all_issues:
        print(f"\n{len(all_issues)} validation issue(s) found:\n")
        for issue in all_issues:
            print(f"  {issue}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
