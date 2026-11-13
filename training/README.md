# Sentinel Intelligence — fine-tuning extension (Part 2)

This extends the Part 1 repository with a dataset pipeline, a LoRA/QLoRA
fine-tuning script, a benchmark harness, and an export pipeline to get a
fine-tuned checkpoint back into Ollama. It does **not** modify Part 1's
CLI, policy engine, executor, or tool adapters — it adds a new
structured-decision contract (`StructuredDecision` in `sentinel/schemas.py`)
and a `ModelRuntime.decide()` method, and adapts onto the existing
`Plan`/`ModelInterpretation` shapes the agent loop already consumes.

## What has and hasn't been verified

Be precise about this, because it matters for anyone picking this up:

**Verified, by actually running it:**
- `training/prepare_dataset.py` — builds train/val/test JSONL from the
  15-example seed set, splits by `scenario_id` (10 scenarios → 6/2/2),
  confirmed zero scenario overlap across splits.
- `training/validate_dataset.py` — confirmed it passes on the real
  prepared splits, and confirmed it actually *catches* problems: an
  unregistered tool reference, a record not marked `synthetic: true`,
  cross-split scenario leakage, and an embedded `password=...`-shaped
  string, each via a deliberately corrupted fixture.
- `training/train_lora.py --dry-run` — resolves config (default and via
  `--config`), prints a memory estimate, and correctly warns when a
  config's estimate exceeds 4 GB (confirmed with both the default 3B
  config, which estimates ~1.5 GB, and an 8B model, which estimates
  ~4.3 GB and triggers the warning).
- `training/evaluate_model.py --self-test` — the scoring logic was
  checked against two deterministic stub deciders: one that returns the
  exact ground-truth label (scores 1.0 on structured-output validity,
  tool selection, and stop-condition accuracy; scores 1.0 on tool
  argument validity / 0.0 policy-violation-rate *restricted to tools
  actually installed in the evaluation environment* — see the caveat
  below) and one that always proposes a nonexistent tool (scores
  policy_violation_rate 1.0, as it should).
- `training/export_model.py modelfile` — writes a real Ollama Modelfile
  and prints the `ollama create` command; checked the generated file's
  content directly.
- `training/export_model.py convert` and `quantize` — confirmed they
  fail loudly with a clear message (exit code 1) when `llama.cpp` isn't
  present at the given path, rather than silently no-op'ing.
- 24 new automated tests (`tests/test_structured_decision.py`,
  `tests/test_training_pipeline.py`, `tests/test_evaluate_model.py`),
  on top of Part 1's 48 — 72 total, all passing.

**Not verified — explicitly, because it genuinely can't be from here:**
- No actual fine-tuning run. The environment this was built in has no
  GPU and its network allowlist doesn't include Hugging Face Hub or PyPI's
  full extras (no `torch`/`transformers`/`peft`/`bitsandbytes` install
  attempted), so `train_lora.py`'s real training path
  (`_run_training`, not `--dry-run`) has not been executed even once.
  Everything in it beyond `--dry-run` is written to the same standards
  as the rest of this codebase but is **unexercised code**, not
  verified-working code.
- No actual `merge`/`convert`/`quantize` run — `merge` needs the same
  missing GPU stack; `convert`/`quantize` need a built `llama.cpp`
  checkout, which wasn't available either.
- No real baseline-vs-fine-tuned benchmark comparison. `evaluate_model.py`
  only has stub-decider self-tests; running it against a real Ollama
  model (baseline or fine-tuned) requires wiring a `Decider` from
  `OllamaRuntime(settings).decide` and a reachable Ollama server, which
  this environment doesn't have.
- The 15-example seed dataset is enough to exercise and validate the
  *pipeline*, not enough to actually fine-tune a model that
  generalizes. A real training run needs on the order of hundreds to
  low thousands of examples per category listed in the brief.

If you pick this up: run `training/evaluate_model.py --self-test` and
the test suite first to confirm the pipeline still works in your
environment, then move to an actual GPU machine for training.

## Hardware reality check (16 GB RAM / 4 GB VRAM laptop)

The brief's hardware target genuinely cannot fine-tune most models in
the 3B-8B range without QLoRA, and even with QLoRA the headroom is
tight. `train_lora.py`'s `estimate_memory_gb()` makes this explicit
rather than hoping it works out — see its docstring for the exact,
challengeable assumptions behind the numbers (4-bit NF4 base weights,
LoRA adapter + AdamW moment buffers, a coarse activation-memory term).
On the default config (`Qwen/Qwen2.5-3B-Instruct`, r=16, batch size 1,
grad accumulation 16, 2048-token sequences):

| Term | Estimate |
|---|---|
| Base weights (4-bit NF4) | ~1.4 GB |
| LoRA adapter (fp32) | ~0.03 GB |
| Optimizer state (AdamW, fp32 moments) | ~0.06 GB |
| Activations (rough, grad-checkpointed) | ~0.02 GB |
| **Total** | **~1.5 GB** |

That leaves real headroom on a 4 GB card — but this is a rough
estimate, not a measurement (the "measured" column doesn't exist here
because training was never actually run — see above). Treat the
activation-memory term especially skeptically; it's the least
predictable part of the arithmetic and the actual PyTorch/CUDA
allocator overhead typically runs higher than the raw tensor math
suggests. An 8B base model already crosses the 4 GB estimate
(`train_lora.py --dry-run --config <an 8B config>` prints ~4.3 GB and
a warning) — don't train that size on the laptop; use the
separate-machine path below.

**Recommended split**, matching the brief's "support training on a
separate machine" requirement:
1. Run `prepare_dataset.py` / `validate_dataset.py` anywhere (no GPU
   needed — they're pure Python + Pydantic).
2. Run `train_lora.py` and `export_model.py merge/convert/quantize` on
   a machine with a real GPU (a cloud instance, a desktop, whatever's
   available) — these need `torch`/`transformers`/`peft`/`bitsandbytes`
   and, for convert/quantize, a built `llama.cpp`.
3. Copy only the final GGUF file and the `Modelfile` back to the
   laptop. `ollama create sentinel-intel:latest -f Modelfile` and set
   `OLLAMA_MODEL=sentinel-intel:latest` in `.env` — the laptop never
   needs the training stack at all.

## Reproducible commands

```bash
# 1. Dataset (no GPU required)
python training/prepare_dataset.py
python training/validate_dataset.py

# 2. Training (needs training/requirements-train.txt + a real GPU)
pip install -r training/requirements-train.txt
python training/train_lora.py --config training/configs/lora_default.yaml --dry-run   # sanity check first
python training/train_lora.py --config training/configs/lora_default.yaml             # actual training

# 3. Evaluation
python training/evaluate_model.py --self-test     # scoring-logic check, no model needed
# To evaluate a real model, write a small script that builds a Decider:
#   from sentinel.model_runtime import OllamaRuntime
#   from sentinel.config import get_settings
#   runtime = OllamaRuntime(get_settings())
#   decider = runtime.decide
# ...and pass it to evaluate_model.evaluate(decider, test_cases, "my-model", policy, assessment)

# 4. Export / deploy
python training/export_model.py merge --adapter-dir training/runs/lora-qwen2.5-3b/final_adapter \
    --base-model Qwen/Qwen2.5-3B-Instruct --out-dir training/runs/merged
python training/export_model.py convert --merged-dir training/runs/merged \
    --llama-cpp-dir /path/to/llama.cpp --out-file training/runs/sentinel-intel.gguf
python training/export_model.py quantize --in-file training/runs/sentinel-intel.gguf \
    --llama-cpp-dir /path/to/llama.cpp --out-file training/runs/sentinel-intel.Q4_K_M.gguf \
    --quant-type Q4_K_M
python training/export_model.py modelfile --gguf-file training/runs/sentinel-intel.Q4_K_M.gguf \
    --model-tag sentinel-intel:latest

# 5. Select it, or roll back
echo "OLLAMA_MODEL=sentinel-intel:latest" >> .env   # fine-tuned
echo "OLLAMA_MODEL=nemotron-mini" >> .env           # roll back to baseline
```

## The structured-decision contract

`sentinel/schemas.py::StructuredDecision` matches Master Prompt 2 §5
exactly:

```json
{
  "assessment_id": "...",
  "observations": [],
  "hypotheses": [],
  "missing_information": [],
  "proposed_action": {"tool": "registered_tool_name.operation", "arguments": {}},
  "rationale": "",
  "expected_evidence": [],
  "stop_condition": "none"
}
```

`StructuredDecision.to_plan_step()` and `.to_interpretation()` adapt a
decision onto the `PlanStep`/`ModelInterpretation` shapes the Part 1
agent loop (`sentinel/agent.py`) already consumes, so the agent loop did
not need to be rewritten to benefit from this — it can call
`model.decide(context)` instead of (or alongside)
`create_plan`/`interpret_result` and adapt the result. **Nothing about
this format authorizes execution**: `proposed_action.tool` is just a
string (`"tool_name.operation"`) until `PolicyEngine.validate_action`
resolves it against the real tool registry and validates `arguments`
against that operation's own Pydantic schema — identical to how a
Part 1 `ProposedAction` is handled. A fine-tuned model proposing
`"nmap.exploit_everything"` is rejected by the policy engine exactly
the same way a hallucinating baseline model would be; this was
confirmed by `evaluate_model.py --self-test`'s "broken decider" case.

## Dataset pipeline notes

- Every one of the 15 seed examples in `data/training/seed_examples.py`
  is explicitly synthetic (invented scenario, invented tool output) and
  `prepare_dataset.py` refuses to build a record from an example that
  doesn't set `synthetic: True`.
- Categories covered (per Master Prompt 2 §3): initial recon planning,
  nmap output interpretation, SMB/FTP/SSH/DNS enumeration interpretation,
  HTTP investigation, distinguishing a possible vulnerability from a
  verified one, pivoting after a contradicted hypothesis, avoiding
  redundant actions, recognizing a suspected scope violation, two stop
  conditions (sufficient evidence, budget exhausted), evidence
  correlation across services, handling an inconclusive check, a
  privilege-escalation lead, and an evidence-grounded finding with
  remediation advice. Not yet covered: a dedicated "candidate
  vulnerability identification from a specific advisory database"
  example distinct from the correlation example, and more volume per
  category — see "not verified" above.
- Splitting is by `scenario_id`, not by example, specifically so
  near-duplicate states from the same fictional machine can't leak
  across train/val/test — `validate_dataset.py`'s cross-split check
  enforces this on every run, not just at prep time.
- `validate_dataset.py` cross-checks every `proposed_action.tool`
  against the **real, current** tool registry
  (`sentinel.tools.base.default_registry()`), not a hardcoded list — if
  Part 1's tool adapters change, this check automatically stays correct.

## Benchmark metrics — how to read them

`evaluate_model.py` computes the metrics from Master Prompt 2 §7. Two
things worth knowing before trusting a number out of it:

1. **`policy_violation_rate` and `tool_argument_validity` depend on
   which tool binaries are actually installed on the machine running
   the evaluation**, because `PolicyEngine.validate_action` genuinely
   checks `tool.is_installed()` — this is correct behavior for the real
   policy engine, but it means a "perfect" decider that proposes a
   100%-correct action can still score a policy violation if (say)
   `smbclient` isn't on PATH. `--self-test` reports this explicitly
   (it prints which tools are installed and runs a second,
   installed-tools-only comparison) rather than hiding it. Run
   evaluation on a machine with every registered tool installed for
   numbers that reflect model quality rather than environment gaps.
2. **`unsupported_claim_rate` is a coarse lexical-overlap heuristic**,
   not a semantic entailment check (see
   `evaluate_model._unsupported_claim_rate`'s docstring). It will miss
   genuinely unsupported claims that happen to reuse a token from the
   context, and will flag genuinely-supported claims that paraphrase
   too far from the source text. Use it to catch egregious fabrication
   (inventing a CVE or hostname that appears nowhere in the evidence),
   not as a precise quality score.

## Known gaps versus the full Master Prompt 2 brief

- No dedicated advisory-correlation database/lookup is implemented —
  the "correlate services with known advisories" category in the
  dataset is illustrative text reasoning, not backed by an actual CVE
  database query tool.
- `evaluate_model.py` does not yet measure inference latency/VRAM for a
  *loaded* model (only wall-clock time around the `Decider` call and,
  if `psutil` is available, process RSS) — GPU memory profiling would
  need `torch.cuda.max_memory_allocated()` wired in against a real
  loaded model, which requires the GPU stack this environment lacked.
- `end-to-end completion rate on reproducible lab scenarios` from §7
  isn't separately implemented; it would need the Part 1 `agent.py`
  loop run to completion against each test scenario rather than scored
  one decision at a time, which `evaluate_model.py` currently does not
  drive.
