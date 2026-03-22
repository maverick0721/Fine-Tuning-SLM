# Fine-Tuning-SLM

Fine-tune small language models with Unsloth + LoRA in either a notebook or CLI workflow.

## What is included

- `finetune.py`: main training pipeline with dataset-format auto-detection.
- `FineTuning.ipynb`: valid Jupyter notebook that calls the same script.
- `requirements.txt`: pinned dependencies for repeatable setup.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support recommended
- Linux environment (or Colab)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## CLI usage

Check options:

```bash
python finetune.py --help
```

Run a sample training job:

```bash
python finetune.py \
	--model-name unsloth/Phi-3-mini-4k-instruct-bnb-4bit \
	--dataset-name sunny199/pharma-instruction-data \
	--split train \
	--epochs 1 \
	--per-device-bs 2 \
	--grad-acc-steps 4 \
	--lr 2e-5 \
	--output-dir phi3_outputs \
	--lora-save-path phi3_pharma_lora
```

Run longer stability training with a fixed step budget:

```bash
python finetune.py \
	--model-name unsloth/Phi-3-mini-4k-instruct-bnb-4bit \
	--dataset-name tatsu-lab/alpaca \
	--split train[:256] \
	--per-device-bs 1 \
	--grad-acc-steps 1 \
	--max-seq-length 512 \
	--max-steps 128 \
	--output-dir stability_outputs \
	--lora-save-path stability_lora
```

Resume from a checkpoint:

```bash
python finetune.py \
	--model-name unsloth/Phi-3-mini-4k-instruct-bnb-4bit \
	--dataset-name tatsu-lab/alpaca \
	--split train[:256] \
	--per-device-bs 1 \
	--grad-acc-steps 1 \
	--max-seq-length 512 \
	--max-steps 160 \
	--output-dir stability_outputs \
	--lora-save-path stability_lora \
	--resume-from-checkpoint stability_outputs/checkpoint-128
```

Run training with automatic post-eval metrics logging (JSONL):

```bash
python finetune.py \
	--model-name unsloth/Phi-3-mini-4k-instruct-bnb-4bit \
	--dataset-name tatsu-lab/alpaca \
	--split train[:128] \
	--max-steps 64 \
	--output-dir autoeval_outputs \
	--lora-save-path autoeval_lora \
	--post-eval \
	--eval-dataset-name tatsu-lab/alpaca \
	--eval-split train[:16]
```

Write metrics as CSV instead:

```bash
python finetune.py \
	--model-name unsloth/Phi-3-mini-4k-instruct-bnb-4bit \
	--dataset-name tatsu-lab/alpaca \
	--split train[:128] \
	--max-steps 64 \
	--output-dir autoeval_outputs \
	--lora-save-path autoeval_lora \
	--post-eval \
	--eval-log-path autoeval_outputs/eval_metrics.csv
```

## Evaluation

Use `evaluate.py` to run prompt generation and/or perplexity checks.

Evaluate your LoRA adapter:

```bash
python evaluate.py \
	--base-model unsloth/Phi-3-mini-4k-instruct-bnb-4bit \
	--adapter-path sanity_lora \
	--dataset-name tatsu-lab/alpaca \
	--split train[:32]
```

## Compare checkpoints from logs

Use `summarize_eval_logs.py` to rank runs from JSONL/CSV metric logs.

Sort by perplexity (best first):

```bash
python summarize_eval_logs.py --root . --sort-by perplexity
```

Sort by average loss and show top 10:

```bash
python summarize_eval_logs.py --root . --sort-by avg_loss --top-k 10
```

## Notebook usage

Open `FineTuning.ipynb` and run cells in order.

## Notes

- By default, the script requires CUDA. Use `--allow-cpu` only for debugging/smoke checks.
- Output LoRA adapters are saved to `--lora-save-path`.
- Use `--save-merged` to also save a merged fp16 model.
- Use `--resume-from-checkpoint` with `--output-dir` to continue interrupted runs.
- Use `--post-eval` to automatically compute perplexity after training and append metrics to JSONL/CSV.
