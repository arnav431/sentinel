#!/usr/bin/env python3
"""Fine-tune a base instruct model on the prepared Sentinel Intelligence
dataset with LoRA/QLoRA, via Hugging Face Transformers + PEFT.

This script is designed to run on a separate, better-resourced machine
(or a cloud GPU) than the 16 GB RAM / 4 GB VRAM laptop described in the
Part 2 brief — see training/README.md for why, and for the exact memory
arithmetic. `export_model.py` is what takes the resulting adapter back
to something the laptop can actually run via Ollama.

It is intentionally conservative about what it assumes is available:
`transformers`/`peft`/`bitsandbytes`/`torch` are imported lazily, inside
`main()`, so `--dry-run` (which only prints the resolved config and a
memory estimate) works even on a machine that doesn't have a GPU stack
installed at all — including the sandbox this script was authored in.

Usage:
    python training/train_lora.py --config training/configs/lora_default.yaml --dry-run
    python training/train_lora.py --config training/configs/lora_default.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclasses.dataclass
class LoraTrainConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    train_file: str = "data/training/splits/train.jsonl"
    val_file: str = "data/training/splits/val.jsonl"
    output_dir: str = "training/runs/lora-qwen2.5-3b"
    seed: int = 1337

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = dataclasses.field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Quantization — 4-bit NF4 QLoRA is the realistic path for a 4 GB
    # VRAM GPU; see README.md's memory arithmetic for why full/fp16
    # fine-tuning of even a 3B model does not fit.
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Training loop
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    gradient_checkpointing: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "LoraTrainConfig":
        import yaml  # lazy: only needed if a config file is actually loaded

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(**{**dataclasses.asdict(cls()), **data})


def estimate_memory_gb(config: LoraTrainConfig, param_count_b: float) -> dict[str, float]:
    """Rough, documented-assumptions memory estimate. Not a substitute for
    measuring actual usage (see README.md's "measured vs. estimated"
    section) — this exists so a user can sanity-check a config against
    their hardware *before* launching a run that will OOM an hour in.

    Assumptions (stated so they're easy to challenge/replace):
      - 4-bit NF4 weights: ~0.5 bytes/param for the frozen base model.
      - LoRA adapter parameters + optimizer state (AdamW, fp32 moments)
        dominate the *trainable* memory; adapter param count is estimated
        from rank r, alpha, and the number of targeted linear layers.
      - Activations are the least predictable term and are given as a
        coarse per-token-per-layer heuristic, scaled by sequence length
        and batch size — treat this term as the most likely to be wrong.
    """
    base_weights_gb = param_count_b * 1e9 * 0.5 / (1024 ** 3)  # 4-bit

    # Toy approximation: 4 target module types x ~2 * r * hidden_dim params
    # each, times an assumed layer count scaling with model size. This is
    # deliberately rough; the only thing it needs to get right is the
    # order of magnitude (LoRA adapters are small relative to the base
    # model, typically tens of millions of params for r=16 on a 3B model).
    assumed_layers = max(16, round(param_count_b * 10))
    hidden_dim_estimate = 2048 if param_count_b <= 4 else 4096
    adapter_params = len(config.lora_target_modules) * 2 * config.lora_r * hidden_dim_estimate * assumed_layers
    adapter_gb = adapter_params * 4 / (1024 ** 3)  # fp32 adapter weights
    optimizer_gb = adapter_gb * 2  # AdamW: 2x fp32 moment buffers

    activations_gb = (
        config.per_device_train_batch_size
        * config.max_seq_length
        * hidden_dim_estimate
        * assumed_layers
        * 2  # bytes, bf16 activations
        / (1024 ** 3)
        * 0.1  # fudge factor for gradient-checkpointing-reduced retention
    )

    total = base_weights_gb + adapter_gb + optimizer_gb + activations_gb
    return {
        "base_weights_4bit_gb": round(base_weights_gb, 2),
        "lora_adapter_gb": round(adapter_gb, 3),
        "optimizer_state_gb": round(optimizer_gb, 3),
        "activations_estimate_gb": round(activations_gb, 2),
        "total_estimate_gb": round(total, 2),
    }


def _param_count_b_from_model_name(name: str) -> float:
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name)
    if match:
        return float(match.group(1))
    return 3.0  # conservative fallback guess if the name doesn't say


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="YAML config overriding defaults in LoraTrainConfig.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and memory estimate; do not import torch/transformers or train.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = LoraTrainConfig.from_yaml(args.config) if args.config else LoraTrainConfig()

    param_b = _param_count_b_from_model_name(config.base_model)
    estimate = estimate_memory_gb(config, param_b)

    print("Resolved training config:")
    print(json.dumps(dataclasses.asdict(config), indent=2))
    print("\nEstimated peak memory (QLoRA 4-bit, rough order-of-magnitude — see docstring):")
    print(json.dumps(estimate, indent=2))

    if estimate["total_estimate_gb"] > 4.0:
        print(
            f"\nWARNING: estimated total ({estimate['total_estimate_gb']} GB) exceeds the 4 GB VRAM "
            f"budget described for the target laptop. This run is a candidate for the separate "
            f"training-machine workflow in README.md, not the laptop GPU.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("\n--dry-run set: not importing torch/transformers/peft, not training.")
        return 0

    return _run_training(config)


def _run_training(config: LoraTrainConfig) -> int:
    """Actual training loop. Imports are local to this function so
    `--dry-run` works without a GPU stack installed at all. NOT executed
    or verified in the sandbox this repository extension was authored
    in (no GPU, no network access to Hugging Face Hub from that
    environment) — see README.md's "what has and hasn't been verified"
    section."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        print(
            f"Missing training dependency: {exc}. Install the optional training extras "
            f"(see training/requirements-train.txt) on the machine that will actually run this.",
            file=sys.stderr,
        )
        return 1

    set_seed(config.seed)
    random.seed(config.seed)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config if config.load_in_4bit else None,
        device_map="auto",
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": config.train_file, "validation": config.val_file},
    )

    def _format(example: dict[str, Any]) -> dict[str, Any]:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        tokens = tokenizer(text, truncation=True, max_length=config.max_seq_length)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(_format, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()

    final_dir = Path(config.output_dir) / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    manifest = {
        "base_model": config.base_model,
        "lora_config": dataclasses.asdict(lora_config) if dataclasses.is_dataclass(lora_config) else {},
        "seed": config.seed,
        "train_file": config.train_file,
        "val_file": config.val_file,
    }
    (final_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Adapter and manifest written to {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
