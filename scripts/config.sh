#!/usr/bin/env bash
#
# Shared configuration sourced by all pipeline scripts.
# Override any of these by exporting them before calling a script, e.g.:
#   MODEL=gemma-2-2b-it QTYPE=BoolQ NUMQ=200 bash scripts/run_data_generation.sh
#
set -euo pipefail

# --- Repo root (parent of this scripts/ directory) -------------------------
export REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# --- Python interpreter -----------------------------------------------------
# Prefer the project virtualenv (../venv) if it exists, otherwise `python3`.
if [ -x "$REPO_ROOT/../venv/bin/python" ]; then
    export PYTHON="$REPO_ROOT/../venv/bin/python"
else
    export PYTHON="${PYTHON:-python3}"
fi

# --- Model / dataset download cache ----------------------------------------
# All HuggingFace models and datasets are downloaded here.
export LLM_COLLAB_CACHE="${LLM_COLLAB_CACHE:-/opt/tiger/AgentMonitor/tmp}"
export HF_HOME="$LLM_COLLAB_CACHE"
mkdir -p "$LLM_COLLAB_CACHE"

# --- GPU selection ----------------------------------------------------------
# Default to a single GPU. The transformers-based model wrappers use
# device_map="auto", which otherwise shards a model across every visible GPU
# and breaks generation. For larger models / tensor-parallel vLLM, override,
# e.g. CUDA_VISIBLE_DEVICES=0,1,2,3.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Multi-GPU vLLM (LLM_COLLAB_TP>1): the eval scripts initialize CUDA in the parent
# before vLLM spawns tensor-parallel workers, so workers must use 'spawn' (fork
# would fail with "Cannot re-initialize CUDA in forked subprocess").
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# --- Local storage for generated debate / preference data ------------------
export STORAGE="${STORAGE:-$REPO_ROOT/storage}"
mkdir -p "$STORAGE"

# --- Model + task defaults --------------------------------------------------
# MODEL     : short name understood by llms.py's LLMHelper (used for inference/data-gen)
# MODEL_HF  : the corresponding full HuggingFace repo id (used by SFT.py / DPO.py)
export MODEL="${MODEL:-gemma-2-2b-it}"
export MODEL_HF="${MODEL_HF:-google/gemma-2-2b-it}"
export QTYPE="${QTYPE:-BoolQ}"
export PROJECT="${PROJECT:-training}"

# model_tag as computed by debate_eval.py: '__'.join(sorted(set(model_names)))
# For a single distinct model this is just $MODEL.
export MODEL_TAG="${MODEL_TAG:-$MODEL}"

# Directory where debate_eval.py reads/writes data for the (target/sup_target)
# stages: <bytenas>/<project>/<Qtype>/<model_tag>/
export DATA_DIR="$STORAGE/$PROJECT/$QTYPE/$MODEL_TAG"

# --- Gated-model token reminder --------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[config] NOTE: HF_TOKEN is not set. Gated models (gemma/llama/mistral)"
    echo "[config]       require a HuggingFace token with access. Run"
    echo "[config]       'huggingface-cli login' or 'export HF_TOKEN=hf_...'."
fi

# questions.py loads datasets from repo-relative dirs (BoolQ/, MMLU/, ...),
# so every stage must run from the repo root.
cd "$REPO_ROOT"
