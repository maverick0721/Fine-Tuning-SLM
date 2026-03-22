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

## Notebook usage

Open `FineTuning.ipynb` and run cells in order.

## Notes

- By default, the script requires CUDA. Use `--allow-cpu` only for debugging/smoke checks.
- Output LoRA adapters are saved to `--lora-save-path`.
- Use `--save-merged` to also save a merged fp16 model.
