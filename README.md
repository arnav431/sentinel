# Sentinel

A terminal-based, agentic penetration-testing assistant for **authorized**
beginner-level Hack The Box labs and vulnerable VMs. It uses
[OpenRouter](https://openrouter.ai) to access a hosted LLM that plans each
step, a deterministic policy engine decides what's actually allowed to
run, and a fixed set of tool adapters (Nmap, HTTP, SMB, FTP, SSH, DNS,
Nuclei, and OWASP ZAP) do the real work. Each adapter exposes a limited
set of validated operations; see `sentinel -t` for registered tools and
their local install status.

Sentinel is **not** a chatbot that prints commands for you to copy. It
executes real tools against real targets you've explicitly put in
scope, records everything in SQLite, and writes it back out as a report.

## What it will not do

- Run against anything you haven't explicitly added to scope.
- Run an intrusive action without your approval (unless you pass
  `--auto-approve`, which you should only do in a fully unattended lab
  script you've already reviewed).
- Attempt credential brute-forcing, exploitation payloads, persistence,
  lateral movement, or data exfiltration. Investigation and *safe
  verification* only.
- Fabricate shells, flags, credentials, or command output. If a tool
  fails or times out, that's what gets reported.

## Requirements

- Python 3.12+.
- The security tools you plan to use installed locally and available on `PATH`.
- An OpenRouter API key. The default model is NVIDIA Nemotron 3.5 Lightning (free); available models and rate limits can change.
- Optional PDF, Markdown, RST, or text references that you have permission to use.

## Install and run

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -e .
```

Install the security tools you want to use, such as Nmap, curl, smbclient, an FTP client, an SSH client, dig, Nuclei, and OWASP ZAP. Ensure the required executables are on `PATH`. Set `OPENROUTER_API_KEY` in your shell or copy `.env.example` to `.env` and set it there.

The repository includes HTB SOC Analyst prerequisites material under
`knowledge/curated/htb-soc-analyst-prerequisites/` (see its README and
LICENSE for source and licensing details). Index it, then add other
authorized references under `knowledge/` and rerun ingestion when they
change:

```bash
sentinel knowledge ingest
sentinel knowledge search "nmap service detection"
sentinel -a 10.10.10.5
```

Starting an assessment asks you to confirm authorization, then saves a new target to scope automatically. You do not need a separate scope command. Use `sentinel -c` to ask beginner-friendly questions about SQL, Bash, TShark syntax, and lab workflows; chat explains commands but does not execute arbitrary shell scripts. TShark itself must be installed locally to analyze packet captures outside chat.

Sentinel stores searchable chunks in a local SQLite database under `data/` and retrieves relevant excerpts for model prompts. This is retrieval-augmented generation (RAG), not model fine-tuning; the OpenRouter model weights remain unchanged. Retrieved excerpts are sent to OpenRouter with the prompt, so do not index private or confidential materials. Supported formats are text, Markdown, RST, and text-based PDFs; scanned PDFs need OCR first.


## CLI reference

| Command | What it does |
|---|---|
| `sentinel knowledge ingest <path>` | Index authorized PDF, Markdown, RST, or text references for retrieval. |
| `sentinel knowledge search <query>` | Search indexed reference excerpts. |
| `sentinel -c` | Interactive chat with the configured OpenRouter model about the current assessment. Never executes tools from chat. |
| `sentinel -t` | List registered tools, install status, versions, operations. |
| `sentinel -a <target> [--rounds N] [--max-iterations N]` | Confirm authorization and run up to 3 feedback-driven tool steps by default; new targets are saved to scope automatically. |
| `sentinel -s` | Status of the current/most recent (or `--assessment <id>`) assessment. |
| `sentinel -f` | List findings with severity, confidence, verification status. |
| `sentinel -r [--format markdown\|json\|both]` | Generate a report. |
| `sentinel -resume <assessment_id>` | Resume a saved assessment; revalidates scope and approvals first. |

Both the flag style above and word-style commands work: `sentinel assess
10.10.10.5`, `sentinel chat`, `sentinel tools`, `sentinel status`,
`sentinel findings`, `sentinel report`, and `sentinel resume <id>`.

## Architecture

```
cli.py        → command surface (assessment, chat, tools, status, findings,
                 reports, resume, and knowledge commands)
agent.py      → the plan → validate → approve → execute → interpret → persist loop
planner.py    → builds model context, turns Plan into actions
model_runtime.py → OpenRouter abstraction (chat/create_plan/interpret_result/summarize_findings)
policy.py     → deterministic scope/risk/argument enforcement — no LLM in this path
scope.py      → authorized-target source of truth
executor.py   → runs validated argv lists; no shell=True, ever
tools/        → one adapter per tool; fixed operations + Pydantic arg schemas
state_store.py→ SQLite persistence (assessments, services, findings, hypotheses, invocations)
findings.py   → hypothesis/finding bookkeeping
reports.py    → Markdown/JSON report generation
```

**Why tool output can never become a command.** Raw stdout/stderr from a
tool is only ever passed to `ModelRuntime.interpret_result()`, whose
return value is parsed into the fixed `ModelInterpretation` schema
(a summary string + hypothesis text). Applying it
(`findings.apply_interpretation`) only ever writes `Hypothesis` rows —
inert data. There is no code path from "text appearing in a service
banner or scan result" to "an argv gets executed." Separately, every
action that *does* get proposed by the model is re-resolved against the
real tool registry and re-validated against that operation's Pydantic
schema in `policy.py` before `executor.py` ever sees it — the model's
raw text is never interpolated into a command line.

Model calls go through the `ModelRuntime` interface
(`chat/create_plan/interpret_result/summarize_findings`), keeping provider
details out of the agent and CLI. The current configuration uses
OpenRouter; see `.env.example` for supported settings.

## Known limitations / untested behavior

- OWASP ZAP (`zap-baseline.py`) and Nuclei must be installed separately
  if you want to use those adapters; their binaries need to be available
  on `PATH`. The same applies to binaries used by the other registered
  adapters.
- The planner's model calls require a valid OpenRouter API key and an
  available configured model; without them, `sentinel -a` records scope/state
  but reports a model error on each iteration.
- Privilege-escalation-lead analysis (parsing authorized
  linPEAS/winPEAS-style enumeration output) is covered by the
  `interpret_result` path's "evidence-driven hypothesis" machinery in
  principle, but no dedicated adapter/parser for those specific
  enumeration script output formats is implemented yet — the hooks
  (`findings.apply_interpretation`, `Hypothesis`/`Finding` schemas) are
  there for an adapter to be added.

## Training experiments

`training/` contains an experimental synthetic dataset pipeline,
fine-tuning scripts, evaluation harness, and Ollama export helpers. The
training README documents what has and has not been verified. This work
is separate from the current assessment runtime; do not assume the
fine-tuned model is used by the CLI.

## Tests

```bash
python -m pip install -e .
python -m pip install pytest pytest-mock
pytest -q
```

The repository includes tests covering scope validation (including rejection
of malicious/invalid targets), policy enforcement (unknown tools,
unknown operations, out-of-scope targets, invalid arguments, approval
gating, the model-cannot-downgrade-risk rule, iteration budget), tool
adapter argument validation and nmap XML parsing, the executor
(timeouts, missing binaries, `shell=True` is never used, output-size
capping), SQLite persistence (including "survives a new process
instance" to model a process restart), report generation, the CLI
surface, that hostile/malicious content embedded in tool output can
never produce a tool invocation, the `StructuredDecision` schema and its
adapters onto Part 1's `Plan`/`ModelInterpretation` shapes, the training
dataset pipeline (synthetic-only enforcement, schema validation,
scenario-leakage-safe splitting, registered-tool cross-checking, secret
scanning), and benchmark scoring against deterministic stub models. Run
`pytest -q` to see the current result in your environment.
