#!/usr/bin/env bash
#
# Stage 3: DPO training on the contrastive pairs produced by
# scripts/run_data_generation.sh.
#
#   SUPPORT=0 (default) : train the ACTOR   from target_c=0 / target_c=1 files
#   SUPPORT=1           : train the CRITIC  from sup_t_c=0 / sup_t_c=1 files
#                         (requires GEN_SUPPORT=1 during data generation)
#
# The merged model is written to $OUTPUT_DIR/<tag>/LORA_DPO_<tag> and can be
# used for inference via scripts/run_eval.sh (SAVED_MODEL=<that path>).
#
# Tunables (env): SUPPORT, EPOCHS, MAX_STEPS, LORA_R, NUM_TRAIN_Q, NUM_EVAL_Q,
#                 MAX_TRAIN, MAX_EVAL, DPO_BATCH, LR
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
DPO_BATCH="${DPO_BATCH:-2}"
LR="${LR:-1.41e-5}"

# RPO regularizer: weight on the NLL/SFT loss added to the DPO loss.
# "None" (default) = pure DPO. Any float (e.g. 0.5, 1.0) enables the RPO term.
RPO_ALPHA="${RPO_ALPHA:-None}"
EXTRA=()
[ "$RPO_ALPHA" != "None" ] && EXTRA+=(--rpo_alpha="$RPO_ALPHA")
# Cap the number of intermediate checkpoints kept on disk (optimizer states are
# large). Set SAVE_TOTAL_LIMIT=1 for experiments to avoid filling the disk.
[ -n "${SAVE_TOTAL_LIMIT:-}" ] && EXTRA+=(--save_total_limit="$SAVE_TOTAL_LIMIT")

if [ "$SUPPORT" = "1" ]; then
    DEBATE_FILES="${DEBATE_FILES:-sup_t_c=0_de,sup_t_c=1_de}"
    PERSONA="${PERSONA:-detail}"
else
    DEBATE_FILES="${DEBATE_FILES:-target_c=0_de,target_c=1_de}"
    PERSONA="${PERSONA:-none}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$STORAGE/dpo_out/$QTYPE/$MODEL}"

$PYTHON DPO.py \
    --model_name_or_path "$MODEL_HF" \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$DPO_BATCH" \
    --gradient_accumulation_steps 2 \
    --output_dir "$OUTPUT_DIR" \
    --lora_r_ "$LORA_R" \
    --logging_steps 10 --eval_steps 50 --save_steps 50 \
    --num_train_epochs "$EPOCHS" --max_steps "$MAX_STEPS" \
    --bf16 True \
    --evaluation_strategy steps --save_strategy steps \
    --json_path "$DATA_DIR/" \
    --debate_files "$DEBATE_FILES" \
    --num_train_questions "$NUM_TRAIN_Q" --num_eval_questions "$NUM_EVAL_Q" \
    --max_train_examples "$MAX_TRAIN" --max_eval_examples "$MAX_EVAL" \
    --support "$SUPPORT" \
    --support_persona "$PERSONA" \
    --max_prompt_length 512 --max_target_length 512 \
    --weight_decay 0.1 \
    --load_best_model_at_end True \
    --project_name LLM_Collab \
    "${EXTRA[@]}"
