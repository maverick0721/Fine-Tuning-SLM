#!/usr/bin/env python3
"""Fine-tune an SLM using Unsloth + LoRA with a reproducible CLI workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class FineTuneConfig:
    # model
    model_name: str
    load_in_4bit: bool = True
    max_seq_length: int = 4096
    dtype: Optional[str] = None  # None = auto

    # dataset
    dataset_name: str = "tatsu-lab/alpaca"
    split: str = "train"
    seed: int = 3407

    # training
    output_dir: str = "outputs"
    lora_save_path: str = "lora_adapters"
    per_device_bs: int = 2
    grad_acc_steps: int = 4
    epochs: int = 1
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    packing: bool = True
    max_steps: int = -1
    resume_from_checkpoint: Optional[str] = None

    # lora
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    use_gc: bool = False

    # save merged model
    save_merged: bool = False
    merged_save_path: str = "merged_fp16_model"


class UnslothFineTuner:
    def __init__(self, cfg: FineTuneConfig):
        self.cfg = cfg
        if self.cfg.target_modules is None:
            self.cfg.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        self.model = None
        self.tokenizer = None

    def _lazy_imports(self):
        from unsloth import FastLanguageModel
        from datasets import load_dataset
        from peft import PeftModel
        from trl import SFTConfig, SFTTrainer

        return load_dataset, PeftModel, SFTTrainer, SFTConfig, FastLanguageModel

    def validate_runtime(self, allow_cpu: bool = False):
        import torch

        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError(
                "CUDA GPU is required by default. Re-run with --allow-cpu to override."
            )

    # Loading Model
    def load_model(self):
        load_dataset, PeftModel, SFTTrainer, SFTConfig, FastLanguageModel = self._lazy_imports()
        _ = load_dataset, SFTTrainer, SFTConfig  # keep static analyzers happy

        print(f"Loading model: {self.cfg.model_name}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.model_name,
            max_seq_length=self.cfg.max_seq_length,
            dtype=self.cfg.dtype,
            load_in_4bit=self.cfg.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cfg.lora_r,
            target_modules=self.cfg.target_modules,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing=self.cfg.use_gc,
            random_state=self.cfg.seed,
        )

        self.model.print_trainable_parameters()
        print("Is PEFT model?", isinstance(self.model, PeftModel))
        print(
            "Device:",
            next(self.model.parameters()).device,
            "| Dtype:",
            next(self.model.parameters()).dtype,
        )

    # Loading Dataset
    def load_dataset(self):
        load_dataset, _, _, _, _ = self._lazy_imports()

        print(f"Loading dataset: {self.cfg.dataset_name} (split={self.cfg.split})")
        try:
            ds = load_dataset(self.cfg.dataset_name, split=self.cfg.split)
        except Exception:
            ds = load_dataset(path=self.cfg.dataset_name, split=self.cfg.split)

        ds = ds.shuffle(seed=self.cfg.seed)
        print(ds)
        print("Columns:", ds.column_names)
        return ds

    # Formatting Helpers
    def _eos(self):
        return self.tokenizer.eos_token or ""

    def _alpaca_prompt(self):
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes "
            "the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )

    def _format_alpaca(self, batch: Dict[str, List[Any]]):
        eos = self._eos()
        prompt = self._alpaca_prompt()
        texts = []
        inp_list = batch.get("input", [""] * len(batch["instruction"]))
        for ins, inp, out in zip(batch["instruction"], inp_list, batch["output"]):
            texts.append(
                prompt.format(
                    instruction=ins or "",
                    input=inp or "",
                    output=out or "",
                )
                + eos
            )
        return {"text": texts}

    def _format_dolly(self, batch: Dict[str, List[Any]]):
        eos = self._eos()
        prompt = self._alpaca_prompt()
        ctx_list = batch.get("context", [""] * len(batch["instruction"]))
        texts = []
        for ins, ctx, resp in zip(batch["instruction"], ctx_list, batch["response"]):
            texts.append(
                prompt.format(
                    instruction=ins or "",
                    input=ctx or "",
                    output=resp or "",
                )
                + eos
            )
        return {"text": texts}

    def _format_sharegpt(self, batch: Dict[str, List[Any]]):
        eos = self._eos()
        texts = []

        if "conversations" in batch:
            conv_key = "conversations"
        elif "messages" in batch:
            conv_key = "messages"
        else:
            raise ValueError("ShareGPT format needs 'conversations' or 'messages' column.")

        for conv in batch[conv_key]:
            if conv is None:
                texts.append(eos)
                continue

            turns = []
            for t in conv:
                if isinstance(t, dict) and "from" in t and "value" in t:
                    role, content = t["from"], t["value"]
                elif isinstance(t, dict) and "role" in t and "content" in t:
                    role, content = t["role"], t["content"]
                else:
                    continue
                turns.append((role, content))

            chat = ""
            for role, content in turns:
                if role in ["human", "user"]:
                    chat += f"### User:\n{content}\n\n"
                else:
                    chat += f"### Assistant:\n{content}\n\n"

            texts.append(chat.strip() + eos)

        return {"text": texts}

    def _format_text(self, batch: Dict[str, List[Any]]):
        eos = self._eos()
        if "text" not in batch:
            raise ValueError("Expected a 'text' column.")
        return {"text": [(t or "") + eos for t in batch["text"]]}

    def _format_pharma_custom(self, batch: Dict[str, List[Any]]):
        eos = self._eos()
        cols = set(batch.keys())

        if {"instruction", "output"}.issubset(cols):
            return self._format_alpaca(batch)

        if {"question", "answer"}.issubset(cols):
            texts = []
            for q, a in zip(batch["question"], batch["answer"]):
                texts.append(f"### Question:\n{q or ''}\n\n### Answer:\n{a or ''}" + eos)
            return {"text": texts}

        if {"prompt", "completion"}.issubset(cols):
            texts = []
            for p, c in zip(batch["prompt"], batch["completion"]):
                texts.append(f"{p or ''}\n{c or ''}" + eos)
            return {"text": texts}

        if {"input", "output"}.issubset(cols):
            prompt = self._alpaca_prompt()
            texts = []
            for i, o in zip(batch["input"], batch["output"]):
                texts.append(
                    prompt.format(
                        instruction="Answer the following:",
                        input=i or "",
                        output=o or "",
                    )
                    + eos
                )
            return {"text": texts}

        raise ValueError(f"Pharma formatter cannot infer columns: {sorted(list(cols))}")

    # Infer dataset format
    def format_dataset(self, ds):
        print("Inferring dataset format...")

        name = self.cfg.dataset_name
        cols = set(ds.column_names)

        if name == "tatsu-lab/alpaca":
            print("Format: ALPACA")
            return ds.map(self._format_alpaca, batched=True, remove_columns=ds.column_names)

        if name == "databricks/databricks-dolly-15k":
            print("Format: DOLLY")
            return ds.map(self._format_dolly, batched=True, remove_columns=ds.column_names)

        if name == "anon8231489123/ShareGPT_Vicuna_unfiltered":
            print("Format: SHAREGPT")
            return ds.map(self._format_sharegpt, batched=True, remove_columns=ds.column_names)

        if name == "OpenAssistant/oasst1":
            print("Format: OASST (auto)")
            if "text" in cols:
                return ds.map(self._format_text, batched=True, remove_columns=ds.column_names)
            return ds.map(self._format_sharegpt, batched=True, remove_columns=ds.column_names)

        if name in ["sunny199/pharma-instruction-data", "pharma_instruction_data"]:
            print("Format: PHARMA (custom)")
            return ds.map(self._format_pharma_custom, batched=True, remove_columns=ds.column_names)

        if "text" in cols:
            print("Format: TEXT (fallback)")
            return ds.map(self._format_text, batched=True, remove_columns=ds.column_names)

        if "conversations" in cols or "messages" in cols:
            print("Format: CHAT (fallback)")
            return ds.map(self._format_sharegpt, batched=True, remove_columns=ds.column_names)

        if {"instruction", "output"}.issubset(cols):
            print("Format: ALPACA-LIKE (fallback)")
            return ds.map(self._format_alpaca, batched=True, remove_columns=ds.column_names)

        raise ValueError(f"Could not infer dataset format. Columns: {sorted(list(cols))}")

    # Train
    def train(self, ds):
        _, _, SFTTrainer, SFTConfig, _ = self._lazy_imports()

        print("Starting training...")
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            packing=self.cfg.packing,
            args=SFTConfig(
                per_device_train_batch_size=self.cfg.per_device_bs,
                gradient_accumulation_steps=self.cfg.grad_acc_steps,
                num_train_epochs=self.cfg.epochs,
                learning_rate=self.cfg.lr,
                warmup_ratio=self.cfg.warmup_ratio,
                optim="adamw_8bit",
                logging_steps=self.cfg.logging_steps,
                seed=self.cfg.seed,
                output_dir=self.cfg.output_dir,
                max_steps=self.cfg.max_steps,
                report_to="none",
            ),
        )
        train_kwargs = {}
        if self.cfg.resume_from_checkpoint:
            print("Resuming from checkpoint:", self.cfg.resume_from_checkpoint)
            train_kwargs["resume_from_checkpoint"] = self.cfg.resume_from_checkpoint
        trainer.train(**train_kwargs)

    # Save
    def save(self):
        print("Saving LoRA adapters to:", self.cfg.lora_save_path)
        self.model.save_pretrained(self.cfg.lora_save_path)
        self.tokenizer.save_pretrained(self.cfg.lora_save_path)

        if self.cfg.save_merged:
            print("Merging LoRA and saving full model to:", self.cfg.merged_save_path)
            merged = self.model.merge_and_unload()
            merged.save_pretrained(self.cfg.merged_save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(self.cfg.merged_save_path)

    def run(self, allow_cpu: bool = False):
        self.validate_runtime(allow_cpu=allow_cpu)
        self.load_model()
        raw_ds = self.load_dataset()
        ds = self.format_dataset(raw_ds)
        print("Sample formatted text:\n", ds["text"][0][:800])
        self.train(ds)
        self.save()
        print("Done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune SLM with Unsloth + LoRA")

    parser.add_argument("--model-name", default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
    parser.add_argument("--dataset-name", default="sunny199/pharma-instruction-data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="phi3_outputs")
    parser.add_argument("--lora-save-path", default="phi3_pharma_lora")
    parser.add_argument("--merged-save-path", default="merged_fp16_model")

    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-bs", type=int, default=2)
    parser.add_argument("--grad-acc-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit model loading")
    parser.add_argument("--no-packing", action="store_true", help="Disable sequence packing")
    parser.add_argument("--save-merged", action="store_true", help="Save merged fp16 model")
    parser.add_argument("--use-gc", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to Trainer checkpoint directory to resume training from.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = FineTuneConfig(
        model_name=args.model_name,
        load_in_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
        dataset_name=args.dataset_name,
        split=args.split,
        seed=args.seed,
        output_dir=args.output_dir,
        lora_save_path=args.lora_save_path,
        per_device_bs=args.per_device_bs,
        grad_acc_steps=args.grad_acc_steps,
        epochs=args.epochs,
        lr=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        packing=not args.no_packing,
        use_gc=args.use_gc,
        save_merged=args.save_merged,
        merged_save_path=args.merged_save_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    trainer = UnslothFineTuner(cfg)
    trainer.run(allow_cpu=args.allow_cpu)


if __name__ == "__main__":
    main()
