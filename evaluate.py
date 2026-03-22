#!/usr/bin/env python3
"""Evaluate a base model + optional LoRA adapter using generation and perplexity."""

from __future__ import annotations

import argparse
import math
from typing import List

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(base_model: str, adapter_path: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype="auto",
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_prompts() -> List[str]:
    return [
        "Explain LoRA fine-tuning in simple terms.",
        "Write a short medication safety disclaimer for patient education content.",
        "Summarize why gradient accumulation is used during training.",
    ]


def run_generation(model, tokenizer, max_new_tokens: int):
    print("\n=== Generation Samples ===")
    prompts = build_prompts()
    for idx, prompt in enumerate(prompts, start=1):
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n[{idx}] Prompt: {prompt}\n{text}\n")


def infer_text_column(columns: List[str]) -> str:
    for key in ["text", "output", "response", "completion"]:
        if key in columns:
            return key
    raise ValueError(f"Could not infer text column from: {columns}")


def run_perplexity(
    model,
    tokenizer,
    dataset_name: str,
    split: str,
    max_samples: int,
    max_length: int,
):
    ds = load_dataset(dataset_name, split=split)
    text_col = infer_text_column(ds.column_names)

    losses = []
    usable = min(max_samples, len(ds))
    print(f"\n=== Perplexity ({usable} samples) ===")

    for i in range(usable):
        text = ds[i][text_col]
        if not text:
            continue

        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with torch.no_grad():
            out = model(**tokens, labels=tokens["input_ids"])
        loss = float(out.loss.item())
        losses.append(loss)

    if not losses:
        raise RuntimeError("No valid samples were evaluated for perplexity.")

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate base model + optional LoRA adapter")
    parser.add_argument(
        "--base-model",
        default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        help="Base model used for training.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to LoRA adapter directory (optional).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per prompt.",
    )
    parser.add_argument(
        "--dataset-name",
        default="tatsu-lab/alpaca",
        help="Dataset for perplexity evaluation.",
    )
    parser.add_argument(
        "--split",
        default="train[:32]",
        help="Dataset split for perplexity evaluation.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=32,
        help="Maximum number of examples for perplexity.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Token length cap per sample for perplexity.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation and only compute perplexity.",
    )
    parser.add_argument(
        "--skip-perplexity",
        action="store_true",
        help="Skip perplexity and only run generation.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.adapter_path)

    if not args.skip_generation:
        run_generation(model, tokenizer, max_new_tokens=args.max_new_tokens)

    if not args.skip_perplexity:
        run_perplexity(
            model,
            tokenizer,
            dataset_name=args.dataset_name,
            split=args.split,
            max_samples=args.max_samples,
            max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
