#!/usr/bin/env bash
#
# Stage 4: evaluate debate accuracy (inference).
#
# By default evaluates the BASE model. To evaluate a DPO/SFT-trained actor,
# point SAVED_MODEL at a merged checkpoint directory (LORA_DPO_* / LORA_SFT_*):
#
#   SAVED_MODEL=storage/dpo_out/BoolQ/gemma-2-2b-it/<tag>/LORA_DPO_<tag> \
#       bash scripts/run_eval.sh
#
# Prints per-round accuracy via debate_eval.py's display_acc().
#
# Tunables (env): NUMQ, ROUNDS, BATCH, SAVED_MODEL, LORA_NAME
#
source "$( dirname "${BASH_SOURCE[0]}" )/config.sh"

NUMQ="${NUMQ:-100}"
ROUNDS="${ROUNDS:-3}"
BATCH="${BATCH:-$NUMQ}"
EVAL_OUT="${EVAL_OUT:-$STORAGE/eval_out/$QTYPE/$MODEL}"

if [ -n "${SAVED_MODEL:-}" ]; then
    # LoRA/merged-model name understood by llms.py (must match the base arch):
    #   gemma  -> V_Gemma-2-LORA   llama2 -> V_Llama-2-LORA
    #   llama3 -> V_Llama-3-LORA   mistral-> V_Mistral-LORA
    ACTOR_MODEL="${LORA_NAME:-V_Gemma-2-LORA}"
    ACTOR_SAVED="$SAVED_MODEL"
    echo "[eval] evaluating TRAINED actor: $SAVED_MODEL"
else
    ACTOR_MODEL="$MODEL"
    ACTOR_SAVED="none"
    echo "[eval] evaluating BASE model: $MODEL"
fi

$PYTHON debate_eval.py --debate_type debate --correct 1 \
    --Qtype "$QTYPE" --num_questions "$NUMQ" --q_start 0 --num_rounds "$ROUNDS" \
    --batch_size "$BATCH" --num_trials 1 \
    --roles none,detail \
    --model_names "$ACTOR_MODEL,$MODEL" \
    --support_list False,True \
    --saved_model_list "$ACTOR_SAVED,none" \
    --temps 0.7,0.7 --max_trys 2 \
    --use_judge False --resume False \
    --bytenas "$STORAGE" --project_name eval --output_dir "$EVAL_OUT" \
    --hold_in_subjects None --hold_out_subjects None
