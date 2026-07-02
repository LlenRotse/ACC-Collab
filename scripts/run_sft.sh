#!/usr/bin/env bash
#
# Optional supervised fine-tuning (SFT) on the "chosen" (correct) responses.
# Same data as DPO, but trains only on the positive side of each pair
# (comparison=False, threshold=0.8 inside data.py).
#
# Tunables (env): SUPPORT, EPOCHS, MAX_STEPS, LORA_R, NUM_TRAIN_Q, NUM_EVAL_Q,
#                 MAX_TRAIN, MAX_EVAL, SFT_BATCH, LR
#
source "$( dirname "${BASH_SOURCE[0]}" )/config.sh"

SUPPORT="${SUPPORT:-0}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
LORA_R="${LORA_R:-128}"
NUM_TRAIN_Q="${NUM_TRAIN_Q:-6000}"
NUM_EVAL_Q="${NUM_EVAL_Q:-1000}"
MAX_TRAIN="${MAX_TRAIN:-20000}"
MAX_EVAL="${MAX_EVAL:-4000}"
SFT_BATCH="${SFT_BATCH:-4}"
LR="${LR:-1.41e-5}"

# Cap intermediate checkpoints on disk (set SAVE_TOTAL_LIMIT=1 for experiments).
EXTRA=()
[ -n "${SAVE_TOTAL_LIMIT:-}" ] && EXTRA+=(--save_total_limit="$SAVE_TOTAL_LIMIT")

if [ "$SUPPORT" = "1" ]; then
    DEBATE_FILES="${DEBATE_FILES:-sup_t_c=0_de,sup_t_c=1_de}"
    PERSONA="${PERSONA:-detail}"
else
    DEBATE_FILES="${DEBATE_FILES:-target_c=0_de,target_c=1_de}"
    PERSONA="${PERSONA:-none}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE/sft_out/$QTYPE/$MODEL}"

$PYTHON SFT.py \
    --model_name_or_path "$MODEL_HF" \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$SFT_BATCH" \
    --auto_find_batch_size True \
    --output_dir "$OUTPUT_DIR" \
    --lora_r_ "$LORA_R" \
    --logging_steps 10 --eval_steps 50 --save_steps 50 \
    --num_train_epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
    --bf16 True \
    --evaluation_strategy steps \
    --json_path "$DATA_DIR/" \
    --debate_files "$DEBATE_FILES" \
    --num_train_questions "$NUM_TRAIN_Q" --num_eval_questions "$NUM_EVAL_Q" \
    --max_train_examples "$MAX_TRAIN" --max_eval_examples "$MAX_EVAL" \
    --support "$SUPPORT" \
    --support_persona "$PERSONA" \
    --weight_decay 0.1 \
    --gradient_accumulation_steps 2 \
    --load_best_model_at_end True \
    "${EXTRA[@]}"
