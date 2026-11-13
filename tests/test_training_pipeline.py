from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

import prepare_dataset  # noqa: E402
import validate_dataset  # noqa: E402


def _load_seed_examples():
    spec = importlib.util.spec_from_file_location(
        "seed_examples", REPO_ROOT / "data/training/seed_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.EXAMPLES


def test_seed_examples_all_declare_synthetic():
    examples = _load_seed_examples()
    assert examples, "seed dataset should not be empty"
    assert all(ex.get("synthetic") is True for ex in examples)


def test_build_record_rejects_non_synthetic_example():
    bad_example = {
        "scenario_id": "x", "category": "y", "synthetic": False,
        "context": {}, "target_decision": {"assessment_id": "a"},
    }
    with pytest.raises(ValueError):
        prepare_dataset.build_record(bad_example)


def test_build_record_validates_target_decision_schema():
    bad_example = {
        "scenario_id": "x", "category": "y", "synthetic": True,
        "context": {}, "target_decision": {"assessment_id": "a", "stop_condition": "not_a_real_value"},
    }
    with pytest.raises(Exception):
        prepare_dataset.build_record(bad_example)


def test_split_by_scenario_has_no_overlap():
    examples = _load_seed_examples()
    train, val, test = prepare_dataset.split_by_scenario(examples, 0.15, 0.15, seed_value=42)
    train_ids = {e["scenario_id"] for e in train}
    val_ids = {e["scenario_id"] for e in val}
    test_ids = {e["scenario_id"] for e in test}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    # every example lands somewhere
    assert len(train) + len(val) + len(test) == len(examples)


def test_split_is_deterministic_given_same_seed():
    examples = _load_seed_examples()
    a = prepare_dataset.split_by_scenario(examples, 0.15, 0.15, seed_value=7)
    b = prepare_dataset.split_by_scenario(examples, 0.15, 0.15, seed_value=7)
    assert [e["scenario_id"] for e in a[0]] == [e["scenario_id"] for e in b[0]]


def test_full_prepare_and_validate_round_trip(tmp_path):
    out_dir = tmp_path / "splits"
    examples = _load_seed_examples()
    train, val, test = prepare_dataset.split_by_scenario(examples, 0.15, 0.15, seed_value=1337)
    for name, split in (("train", train), ("val", val), ("test", test)):
        records = [prepare_dataset.build_record(ex) for ex in split]
        prepare_dataset.write_jsonl(records, out_dir / f"{name}.jsonl")

    registered = validate_dataset._registered_pairs()
    all_issues = []
    for name in ("train", "val", "test"):
        issues, _ = validate_dataset.validate_file(out_dir / f"{name}.jsonl", registered)
        all_issues.extend(issues)
    assert not all_issues, [str(i) for i in all_issues]


def test_validate_dataset_catches_unregistered_tool(tmp_path):
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Assessment state (JSON):\n{}"},
            {"role": "assistant", "content": json.dumps({
                "assessment_id": "a",
                "proposed_action": {"tool": "nmap.not_a_real_op", "arguments": {}},
                "stop_condition": "none",
            })},
        ],
        "meta": {"scenario_id": "s1", "category": "c", "synthetic": True},
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record) + "\n")
    registered = validate_dataset._registered_pairs()
    issues, _ = validate_dataset.validate_file(path, registered)
    assert any("not a registered tool.operation pair" in str(i) for i in issues)


def test_validate_dataset_catches_missing_synthetic_flag(tmp_path):
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Assessment state (JSON):\n{}"},
            {"role": "assistant", "content": json.dumps({"assessment_id": "a", "stop_condition": "none"})},
        ],
        "meta": {"scenario_id": "s1", "category": "c", "synthetic": False},
    }
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(record) + "\n")
    registered = validate_dataset._registered_pairs()
    issues, _ = validate_dataset.validate_file(path, registered)
    assert any("not marked synthetic" in str(i) for i in issues)


def test_validate_dataset_catches_cross_split_leakage(tmp_path):
    def rec(scenario_id):
        return {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Assessment state (JSON):\n{}"},
                {"role": "assistant", "content": json.dumps({"assessment_id": "a", "stop_condition": "none"})},
            ],
            "meta": {"scenario_id": scenario_id, "category": "c", "synthetic": True},
        }

    out_dir = tmp_path / "leaky_splits"
    out_dir.mkdir()
    (out_dir / "train.jsonl").write_text(json.dumps(rec("shared-scenario")) + "\n")
    (out_dir / "val.jsonl").write_text(json.dumps(rec("shared-scenario")) + "\n")
    (out_dir / "test.jsonl").write_text(json.dumps(rec("only-in-test")) + "\n")

    registered = validate_dataset._registered_pairs()
    scenario_ids_by_split = {}
    for name in ("train", "val", "test"):
        _, scenario_ids = validate_dataset.validate_file(out_dir / f"{name}.jsonl", registered)
        scenario_ids_by_split[name] = scenario_ids

    overlap = scenario_ids_by_split["train"] & scenario_ids_by_split["val"]
    assert overlap == {"shared-scenario"}


def test_validate_dataset_flags_possible_secret(tmp_path):
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "password: hunter22222"},
            {"role": "assistant", "content": json.dumps({"assessment_id": "a", "stop_condition": "none"})},
        ],
        "meta": {"scenario_id": "s1", "category": "c", "synthetic": True},
    }
    path = tmp_path / "secret.jsonl"
    path.write_text(json.dumps(record) + "\n")
    registered = validate_dataset._registered_pairs()
    issues, _ = validate_dataset.validate_file(path, registered)
    assert any("possible secret" in str(i) for i in issues)
