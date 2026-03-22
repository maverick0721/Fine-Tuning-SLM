#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="quick"
SKIP_INSTALL=0
MAX_STEPS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_demo.sh [--quick|--full] [--skip-install] [--max-steps N]"
            exit 1
            ;;
    esac
done

if [[ "$MODE" == "quick" ]]; then
    TRAIN_SPLIT="train[:32]"
    EVAL_SPLIT="train[:8]"
    TRAIN_STEPS=8
    EVAL_SAMPLES=8
    MAX_NEW_TOKENS=64
else
    TRAIN_SPLIT="train[:128]"
    EVAL_SPLIT="train[:32]"
    TRAIN_STEPS=64
    EVAL_SAMPLES=32
    MAX_NEW_TOKENS=128
fi

if [[ -n "$MAX_STEPS" ]]; then
    TRAIN_STEPS="$MAX_STEPS"
fi

MODEL_NAME="unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
DATASET_NAME="tatsu-lab/alpaca"

RUN_ID="demo_$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="demo_runs/${RUN_ID}"
TRAIN_DIR="${RUN_DIR}/train"
LORA_DIR="${RUN_DIR}/lora"
EVAL_LOG_PATH="${RUN_DIR}/eval_metrics.jsonl"
EVAL_REPORT_PATH="${RUN_DIR}/evaluate_report.txt"
RANKING_PATH="${RUN_DIR}/ranking.txt"

mkdir -p "$RUN_DIR"

echo "========================================"
echo " Fine-Tuning-SLM Demo Launcher"
echo "========================================"
echo "Mode: $MODE"
echo "Run ID: $RUN_ID"
echo "Run directory: $RUN_DIR"
echo

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. GPU training may not work in this environment."
fi
echo

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    echo "Skipping dependency installation (--skip-install)."
fi

echo
echo "[1/3] Training + automatic post-eval logging"
python finetune.py \
    --model-name "$MODEL_NAME" \
    --dataset-name "$DATASET_NAME" \
    --split "$TRAIN_SPLIT" \
    --per-device-bs 1 \
    --grad-acc-steps 1 \
    --max-seq-length 512 \
    --max-steps "$TRAIN_STEPS" \
    --output-dir "$TRAIN_DIR" \
    --lora-save-path "$LORA_DIR" \
    --post-eval \
    --eval-dataset-name "$DATASET_NAME" \
    --eval-split "$EVAL_SPLIT" \
    --eval-max-samples "$EVAL_SAMPLES" \
    --eval-max-length 512 \
    --eval-log-path "$EVAL_LOG_PATH"

echo
echo "[2/3] Standalone evaluation (generation + perplexity)"
python evaluate.py \
    --base-model "$MODEL_NAME" \
    --adapter-path "$LORA_DIR" \
    --dataset-name "$DATASET_NAME" \
    --split "$EVAL_SPLIT" \
    --max-samples "$EVAL_SAMPLES" \
    --max-length 512 \
    --max-new-tokens "$MAX_NEW_TOKENS" | tee "$EVAL_REPORT_PATH"

echo
echo "[3/3] Ranking logs"
python summarize_eval_logs.py \
    --root "$RUN_DIR" \
    --sort-by perplexity \
    --top-k 10 | tee "$RANKING_PATH"

echo
echo "========================================"
echo " Demo completed successfully"
echo "========================================"
echo "Artifacts:"
echo "- Adapter: $LORA_DIR"
echo "- Trainer output: $TRAIN_DIR"
echo "- Eval metrics log: $EVAL_LOG_PATH"
echo "- Eval report: $EVAL_REPORT_PATH"
echo "- Ranking table: $RANKING_PATH"
echo
echo "Recruiter/interviewer talking points:"
echo "1) End-to-end LoRA fine-tuning pipeline runs from one command."
echo "2) Post-training eval metrics are logged automatically for tracking."
echo "3) Runs are comparable through a ranking utility for reproducible reporting."
