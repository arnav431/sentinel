#!/usr/bin/env python3
"""Take a trained LoRA adapter (from train_lora.py) and produce something
the laptop-side Sentinel can actually load through Ollama.

Ollama does not load raw Hugging Face checkpoints or PEFT adapters
directly — it loads GGUF files (optionally quantized) via a Modelfile.
So this is a real pipeline with real incompatible-format steps, not a
single conversion call:

    1. merge   : merge the LoRA adapter into the base model's weights,
                 producing a single standalone Transformers checkpoint
                 (safetensors). Needs `peft`/`transformers`/`torch`.
    2. convert : convert that merged checkpoint to GGUF using
                 llama.cpp's `convert_hf_to_gguf.py`. This step shells
                 out to a llama.cpp checkout and is NOT a Python import —
                 see `--llama-cpp-dir`.
    3. quantize: (optional) quantize the GGUF to a smaller format (e.g.
                 Q4_K_M) with llama.cpp's `llama-quantize` binary, for a
                 closer fit to a 4 GB VRAM / 16 GB RAM laptop.
    4. modelfile: write an Ollama `Modelfile` pointing at the resulting
                 GGUF and print the `ollama create` command.

Each step is a separate subprocess invocation you can run independently
and inspect. Nothing here was executed end-to-end in the sandbox this
script was authored in (no GPU, no llama.cpp build, no network path to
Hugging Face Hub or llama.cpp releases from that environment's allowed
domains) — see README.md for exactly what was and wasn't verified.

Usage:
    python training/export_model.py merge --adapter-dir training/runs/lora-qwen2.5-3b/final_adapter \\
        --base-model Qwen/Qwen2.5-3B-Instruct --out-dir training/runs/merged
    python training/export_model.py convert --merged-dir training/runs/merged \\
        --llama-cpp-dir ../llama.cpp --out-file training/runs/sentinel-intel.gguf
    python training/export_model.py quantize --in-file training/runs/sentinel-intel.gguf \\
        --llama-cpp-dir ../llama.cpp --out-file training/runs/sentinel-intel.Q4_K_M.gguf \\
        --quant-type Q4_K_M
    python training/export_model.py modelfile --gguf-file training/runs/sentinel-intel.Q4_K_M.gguf \\
        --out-file training/runs/Modelfile --model-tag sentinel-intel:latest
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT = (
    "You are Sentinel, an assistant that helps a student investigate an "
    "explicitly authorized beginner-level lab machine (e.g. Hack The Box). "
    "You never fabricate tool output, credentials, flags, or shell access. "
    "You only ever propose actions; you cannot execute anything yourself."
)


def cmd_merge(args: argparse.Namespace) -> int:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"Missing dependency for merge: {exc}. See training/requirements-train.txt.", file=sys.stderr)
        return 1

    print(f"Loading base model {args.base_model} ...")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Applying adapter from {args.adapter_dir} ...")
    merged = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = merged.merge_and_unload()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.out_dir)

    manifest_path = Path(args.adapter_dir) / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    (args.out_dir / "export_manifest.json").write_text(json.dumps({
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "training_manifest": manifest,
        "step": "merge",
    }, indent=2))

    print(f"Merged checkpoint written to {args.out_dir}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    convert_script = args.llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(
            f"Expected llama.cpp's convert_hf_to_gguf.py at {convert_script} but it was not found. "
            f"Clone/build llama.cpp first (https://github.com/ggml-org/llama.cpp) and pass its path "
            f"via --llama-cpp-dir. This step was not run in the authoring sandbox — see README.md.",
            file=sys.stderr,
        )
        return 1

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(convert_script),
        str(args.merged_dir),
        "--outfile", str(args.out_file),
        "--outtype", args.out_type,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def cmd_quantize(args: argparse.Namespace) -> int:
    quantize_bin = args.llama_cpp_dir / "llama-quantize"
    if not quantize_bin.exists():
        print(
            f"Expected llama.cpp's built 'llama-quantize' binary at {quantize_bin} but it was not "
            f"found. Build llama.cpp (cmake --build .) first. This step was not run in the "
            f"authoring sandbox — see README.md.",
            file=sys.stderr,
        )
        return 1

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(quantize_bin), str(args.in_file), str(args.out_file), args.quant_type]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def cmd_modelfile(args: argparse.Namespace) -> int:
    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT
    modelfile_text = (
        f"FROM {args.gguf_file}\n\n"
        f'SYSTEM """{system_prompt}"""\n\n'
        f"PARAMETER temperature {args.temperature}\n"
        f"PARAMETER num_ctx {args.num_ctx}\n"
    )
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(modelfile_text, encoding="utf-8")
    print(f"Modelfile written to {args.out_file}\n")
    print("To load it into Ollama:")
    print(f"  ollama create {args.model_tag} -f {args.out_file}")
    print(f"\nTo roll back to the baseline model, just set OLLAMA_MODEL back to the baseline tag "
          f"(e.g. nemotron-mini) in .env — nothing else needs to change; both are selected the "
          f"same way through sentinel/config.py.")
    print(f"\nTo select this fine-tuned model in Sentinel: set OLLAMA_MODEL={args.model_tag} in .env.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)

    p_merge = sub.add_parser("merge", help="Merge a LoRA adapter into the base model's weights.")
    p_merge.add_argument("--adapter-dir", type=Path, required=True)
    p_merge.add_argument("--base-model", type=str, required=True)
    p_merge.add_argument("--out-dir", type=Path, required=True)
    p_merge.set_defaults(func=cmd_merge)

    p_convert = sub.add_parser("convert", help="Convert a merged HF checkpoint to GGUF via llama.cpp.")
    p_convert.add_argument("--merged-dir", type=Path, required=True)
    p_convert.add_argument("--llama-cpp-dir", type=Path, required=True)
    p_convert.add_argument("--out-file", type=Path, required=True)
    p_convert.add_argument("--out-type", default="f16", choices=["f16", "f32", "bf16"])
    p_convert.set_defaults(func=cmd_convert)

    p_quant = sub.add_parser("quantize", help="Quantize a GGUF file via llama.cpp's llama-quantize.")
    p_quant.add_argument("--in-file", type=Path, required=True)
    p_quant.add_argument("--llama-cpp-dir", type=Path, required=True)
    p_quant.add_argument("--out-file", type=Path, required=True)
    p_quant.add_argument("--quant-type", default="Q4_K_M")
    p_quant.set_defaults(func=cmd_quantize)

    p_mf = sub.add_parser("modelfile", help="Write an Ollama Modelfile and print the load command.")
    p_mf.add_argument("--gguf-file", type=Path, required=True)
    p_mf.add_argument("--out-file", type=Path, default=REPO_ROOT / "training/runs/Modelfile")
    p_mf.add_argument("--model-tag", default="sentinel-intel:latest")
    p_mf.add_argument("--system-prompt", default=None)
    p_mf.add_argument("--temperature", type=float, default=0.2)
    p_mf.add_argument("--num-ctx", type=int, default=4096)
    p_mf.set_defaults(func=cmd_modelfile)

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
