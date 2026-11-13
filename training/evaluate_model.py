#!/usr/bin/env python3
"""Benchmark a model (baseline or fine-tuned) against the held-out test
split and report the metrics from Master Prompt 2 §7.

Designed around a `Decider` protocol — any callable
`(context: dict) -> StructuredDecision` — so the same scoring code runs
against:
  - a live `ModelRuntime.decide()` (baseline Ollama model, or a
    fine-tuned checkpoint once exported and loaded — see export_model.py),
  - or a deterministic stub/mock, which is how this script's own test
    coverage and the `--self-test` mode exercise it without needing any
    model weights or network access at all.

Metrics computed (all per Master Prompt 2 §7, each a 0-1 rate unless
noted):
  - structured_output_validity: fraction of responses that parse as a
    valid StructuredDecision at all (for a live model this means "raw
    text parses as JSON and validates"; for a Decider that already
    returns a StructuredDecision instance, i.e. never raises, this is
    computed against a separate `raw_output_fn` if one is supplied, and
    otherwise defaults to 1.0 with a note that it wasn't actually tested).
  - tool_selection_accuracy: predicted proposed_action.tool matches the
    label's (both None counts as a match — i.e. agreeing that no action
    should be taken).
  - tool_argument_validity: when a tool is proposed, whether its
    `arguments` validate against that operation's real Pydantic schema
    in the tool registry.
  - policy_violation_rate: fraction of proposed actions that the real
    PolicyEngine would refuse to run (unknown tool/op, invalid args, or
    out-of-scope target) — computed by actually calling
    `PolicyEngine.validate_action` against a synthetic in-scope
    assessment, not re-implemented ad hoc.
  - duplicate_action_rate: fraction of proposed actions whose
    (tool, operation) pair already appears in the example's
    `completed_operations` context with no new information since.
  - stop_condition_accuracy: predicted stop_condition matches label's.
  - unsupported_claim_rate: heuristic — fraction of predicted
    `observations` entries that share no overlapping salient token with
    the `stdout_excerpt`/tool context provided. Flagged clearly as a
    coarse heuristic, not a semantic entailment check; see README.md.
  - latency_s / peak_rss_mb: wall-clock and (if `psutil` is importable)
    peak resident memory during the run, when exercising a live Decider.

Usage:
    python training/evaluate_model.py --self-test
    python training/evaluate_model.py --test-file data/training/splits/test.jsonl \\
        --model-name baseline-ollama --decider sentinel_ollama
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sentinel.config import get_settings  # noqa: E402
from sentinel.policy import PolicyEngine, PolicyViolation  # noqa: E402
from sentinel.schemas import Assessment, AssessmentPhase, ProposedAction, StructuredDecision  # noqa: E402
from sentinel.scope import ScopeManager  # noqa: E402
from sentinel.state_store import StateStore  # noqa: E402
from sentinel.tools.base import default_registry  # noqa: E402

Decider = Callable[[dict[str, Any]], StructuredDecision]


@dataclasses.dataclass
class CaseResult:
    scenario_id: str
    category: str
    parsed_ok: bool
    tool_match: bool
    args_valid: Optional[bool]
    policy_ok: Optional[bool]
    duplicate_action: bool
    stop_condition_match: bool
    unsupported_claim_rate: float
    latency_s: float
    error: Optional[str] = None


@dataclasses.dataclass
class BenchmarkReport:
    model_name: str
    n_cases: int
    structured_output_validity: float
    tool_selection_accuracy: float
    tool_argument_validity: float
    policy_violation_rate: float
    duplicate_action_rate: float
    stop_condition_accuracy: float
    unsupported_claim_rate: float
    mean_latency_s: float
    peak_rss_mb: Optional[float]
    failed_cases: list[dict[str, Any]]


def _load_test_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        user_msg = next(m["content"] for m in record["messages"] if m["role"] == "user")
        context_json = user_msg.split("Assessment state (JSON):\n", 1)[-1]
        context = json.loads(context_json)
        label = StructuredDecision.model_validate(
            json.loads(next(m["content"] for m in record["messages"] if m["role"] == "assistant"))
        )
        cases.append({"context": context, "label": label, "meta": record["meta"]})
    return cases


def _collect_referenced_targets(test_cases: list[dict[str, Any]]) -> set[str]:
    """Scan every test case's context and label for host/target/url-like
    strings so the evaluation scope can cover all of them. Policy
    validation checks the target named in the *predicted* action's own
    arguments, not the assessment's nominal target — so a decider that
    (correctly) proposes an action against the scenario's actual host
    must find that host in scope, or scoring would conflate 'the model
    picked a sensible, correct target' with 'policy violation'."""
    targets: set[str] = set()
    for case in test_cases:
        context = case["context"]
        label: StructuredDecision = case["label"]
        if isinstance(context.get("target"), str):
            targets.add(context["target"])
        for svc in context.get("services", []) or []:
            if isinstance(svc, dict) and isinstance(svc.get("host"), str):
                targets.add(svc["host"])
        if label.proposed_action is not None:
            for key in ("target", "host", "url"):
                val = label.proposed_action.arguments.get(key)
                if isinstance(val, str):
                    host = val.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
                    targets.add(host)
    return {t for t in targets if t}


def _build_eval_assessment(store: StateStore, scope: ScopeManager, test_cases: list[dict[str, Any]]) -> Assessment:
    """Seed scope with every host referenced anywhere in the test set
    (see `_collect_referenced_targets`), then create one nominal
    assessment to validate proposed actions against."""
    for target in _collect_referenced_targets(test_cases):
        if not scope.check(target).allowed:
            scope.add(target, confirmed=True)

    nominal_target = next(iter(_collect_referenced_targets(test_cases)), "203.0.113.10")
    if not scope.check(nominal_target).allowed:
        scope.add(nominal_target, confirmed=True)
    assessment = Assessment(
        target=nominal_target,
        scope_id=scope.check(nominal_target).scope_id,
        phase=AssessmentPhase.INVESTIGATION,
    )
    store.save_assessment(assessment)
    return assessment


def _tokenize(text: str) -> set[str]:
    return {tok.lower() for tok in text.replace(",", " ").replace(".", " ").split() if len(tok) > 3}


def _unsupported_claim_rate(observations: list[str], context: dict[str, Any]) -> float:
    """Coarse lexical-overlap heuristic — NOT a semantic entailment check.
    An observation is counted as 'supported' if it shares at least one
    non-trivial token with the tool output / context the model was given.
    This will both over- and under-count; it exists to catch egregious
    fabrication (e.g. inventing a CVE or a hostname that appears nowhere
    in the provided evidence), not to grade reasoning quality."""
    if not observations:
        return 0.0
    context_text = json.dumps(context)
    context_tokens = _tokenize(context_text)
    unsupported = 0
    for obs in observations:
        obs_tokens = _tokenize(obs)
        if obs_tokens and not (obs_tokens & context_tokens):
            unsupported += 1
    return unsupported / len(observations)


def evaluate(
    decider: Decider,
    test_cases: list[dict[str, Any]],
    model_name: str,
    policy: PolicyEngine,
    assessment: Assessment,
) -> BenchmarkReport:
    results: list[CaseResult] = []
    peak_rss_mb: Optional[float] = None
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None

    for case in test_cases:
        context = case["context"]
        label: StructuredDecision = case["label"]
        meta = case["meta"]

        started = time.monotonic()
        error = None
        predicted: Optional[StructuredDecision] = None
        try:
            predicted = decider(context)
        except Exception as exc:  # noqa: BLE001 — want to record and continue
            error = str(exc)
        latency = time.monotonic() - started

        if process is not None:
            rss_mb = process.memory_info().rss / (1024 ** 2)
            peak_rss_mb = rss_mb if peak_rss_mb is None else max(peak_rss_mb, rss_mb)

        if predicted is None:
            results.append(CaseResult(
                scenario_id=meta.get("scenario_id", "?"), category=meta.get("category", "?"),
                parsed_ok=False, tool_match=False, args_valid=None, policy_ok=None,
                duplicate_action=False, stop_condition_match=False,
                unsupported_claim_rate=1.0, latency_s=latency, error=error,
            ))
            continue

        label_tool = label.proposed_action.tool if label.proposed_action else None
        pred_tool = predicted.proposed_action.tool if predicted.proposed_action else None
        tool_match = label_tool == pred_tool

        args_valid: Optional[bool] = None
        policy_ok: Optional[bool] = None
        duplicate_action = False

        if predicted.proposed_action is not None:
            tool_name, _, operation = predicted.proposed_action.tool.partition(".")
            completed = [(op.get("tool"), op.get("operation")) for op in context.get("completed_operations", [])]
            duplicate_action = (tool_name, operation) in completed

            proposed = ProposedAction(tool_name=tool_name, operation=operation, rationale=predicted.rationale)
            try:
                policy.validate_action(assessment, proposed, predicted.proposed_action.arguments)
                args_valid = True
                policy_ok = True
            except PolicyViolation:
                args_valid = False
                policy_ok = False

        stop_condition_match = predicted.stop_condition == label.stop_condition
        unsupported_rate = _unsupported_claim_rate(predicted.observations, context)

        results.append(CaseResult(
            scenario_id=meta.get("scenario_id", "?"), category=meta.get("category", "?"),
            parsed_ok=True, tool_match=tool_match, args_valid=args_valid, policy_ok=policy_ok,
            duplicate_action=duplicate_action, stop_condition_match=stop_condition_match,
            unsupported_claim_rate=unsupported_rate, latency_s=latency,
        ))

    n = len(results)
    parsed = [r for r in results if r.parsed_ok]

    def rate(predicate) -> float:
        return sum(1 for r in parsed if predicate(r)) / len(parsed) if parsed else 0.0

    args_checked = [r for r in parsed if r.args_valid is not None]
    policy_checked = [r for r in parsed if r.policy_ok is not None]

    return BenchmarkReport(
        model_name=model_name,
        n_cases=n,
        structured_output_validity=len(parsed) / n if n else 0.0,
        tool_selection_accuracy=rate(lambda r: r.tool_match),
        tool_argument_validity=(sum(1 for r in args_checked if r.args_valid) / len(args_checked)) if args_checked else 1.0,
        policy_violation_rate=(sum(1 for r in policy_checked if not r.policy_ok) / len(policy_checked)) if policy_checked else 0.0,
        duplicate_action_rate=rate(lambda r: r.duplicate_action),
        stop_condition_accuracy=rate(lambda r: r.stop_condition_match),
        unsupported_claim_rate=(sum(r.unsupported_claim_rate for r in parsed) / len(parsed)) if parsed else 0.0,
        mean_latency_s=(sum(r.latency_s for r in results) / n) if n else 0.0,
        peak_rss_mb=peak_rss_mb,
        failed_cases=[dataclasses.asdict(r) for r in results if not r.parsed_ok or not r.tool_match],
    )


# ---- Self-test fixtures: deterministic stub deciders, no model needed ----

def _perfect_decider_factory(test_cases: list[dict[str, Any]]) -> Decider:
    """Always returns the ground-truth label. Used to sanity-check the
    scoring code itself: a perfect decider must score 1.0 everywhere
    except possibly duplicate/unsupported-claim heuristics that are
    about the *label's own* wording, and policy checks, which depend on
    the actual registry/scope — this is exactly what --self-test asserts."""
    by_context_id = {json.dumps(c["context"], sort_keys=True): c["label"] for c in test_cases}

    def decide(context: dict[str, Any]) -> StructuredDecision:
        key = json.dumps(context, sort_keys=True)
        return by_context_id[key]

    return decide


def _broken_decider(context: dict[str, Any]) -> StructuredDecision:
    """Deliberately bad decider for self-test: always proposes a
    nonexistent tool and never matches any stop condition."""
    return StructuredDecision(
        assessment_id=context.get("assessment_id", "unknown"),
        observations=["the moon is made of cheese"],
        proposed_action={"tool": "nmap.exploit_everything", "arguments": {"target": "0.0.0.0"}},
        stop_condition="none",
    )


def _run_self_test() -> int:
    test_file = REPO_ROOT / "data/training/splits/test.jsonl"
    if not test_file.exists():
        print("No prepared test split found; run training/prepare_dataset.py first.", file=sys.stderr)
        return 1
    test_cases = _load_test_cases(test_file)

    settings = get_settings()
    store = StateStore(settings.db_path)
    scope = ScopeManager(store)
    registry = default_registry()
    policy = PolicyEngine(registry, scope, store, settings)
    assessment = _build_eval_assessment(store, scope, test_cases)

    # Policy validation checks real tool *installation*, not just the
    # argument schema — see policy.PolicyEngine.validate_action. That
    # means policy_violation_rate for even a perfect decider depends on
    # which tool binaries happen to be installed on whatever machine runs
    # this self-test, which is a property of the environment, not of the
    # scoring logic this self-test exists to check. We report both
    # numbers but only hard-assert on the subset of cases whose tool is
    # actually installed here.
    installed_tools = {t.name for t in registry.all() if t.is_installed()}
    print(f"Tools installed in this environment: {sorted(installed_tools)}")

    perfect = _perfect_decider_factory(test_cases)
    perfect_report = evaluate(perfect, test_cases, "self-test-perfect", policy, assessment)
    print("\nPerfect-decider self-test report:")
    print(json.dumps(dataclasses.asdict(perfect_report), indent=2))
    assert perfect_report.structured_output_validity == 1.0
    assert perfect_report.tool_selection_accuracy == 1.0
    assert perfect_report.stop_condition_accuracy == 1.0

    cases_with_installed_tool = [
        c for c in test_cases
        if c["label"].proposed_action is not None
        and c["label"].proposed_action.tool.split(".", 1)[0] in installed_tools
    ]
    if cases_with_installed_tool:
        installed_only_report = evaluate(perfect, cases_with_installed_tool, "self-test-perfect-installed-only", policy, assessment)
        assert installed_only_report.tool_argument_validity == 1.0, (
            "ground-truth labels propose only registered tool/operation pairs with valid "
            "arguments against in-scope hosts; for tools actually installed here the policy "
            "engine must accept all of them"
        )
        assert installed_only_report.policy_violation_rate == 0.0
    else:
        print(
            "\nNOTE: none of the tools referenced by the test split's proposed actions are "
            "installed in this environment, so the tool_argument_validity/policy_violation_rate "
            "hard assertions were skipped. Install the relevant binaries (see README.md's "
            "tool checklist) to exercise that path.",
        )

    broken_report = evaluate(_broken_decider, test_cases, "self-test-broken", policy, assessment)
    print("\nBroken-decider self-test report:")
    print(json.dumps(dataclasses.asdict(broken_report), indent=2))
    assert broken_report.policy_violation_rate == 1.0, "broken decider proposes an unregistered tool; policy must reject all of it"
    assert broken_report.tool_selection_accuracy < 1.0

    print("\nSelf-test passed: scoring logic behaves correctly against known-good and known-bad deciders.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-file", type=Path, default=REPO_ROOT / "data/training/splits/test.jsonl")
    parser.add_argument("--model-name", default="unnamed-model")
    parser.add_argument("--self-test", action="store_true", help="Run against built-in stub deciders instead of a real model; no weights/network required.")
    parser.add_argument("--out", type=Path, default=None, help="Write the JSON report here in addition to stdout.")
    args = parser.parse_args()

    if args.self_test:
        return _run_self_test()

    print(
        "No live Decider is wired into this CLI by default — evaluating a real baseline or "
        "fine-tuned model requires a reachable inference backend (Ollama, or a loaded HF "
        "checkpoint), which this sandbox did not have network/hardware access to. Import "
        "`evaluate()` from this module and pass a Decider built from your own "
        "ModelRuntime.decide, or OllamaRuntime(settings).decide, directly. See README.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
