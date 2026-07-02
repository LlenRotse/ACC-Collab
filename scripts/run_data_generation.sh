#!/usr/bin/env bash
#
# Stage 1 + 2: generate DPO training data.
#
#   (1) debate     : run the actor/critic debate to produce base trajectories
#   (2) target     : elicit answers toward the CORRECT (c=1, "chosen") and the
#                    INCORRECT (c=0, "rejected") answer -> contrastive actor pairs
#   (3) sup_target : score critic responses by resampling the actor            (optional)
#
# Output files land in $DATA_DIR (see scripts/config.sh):
#   roles=no-de_r=<R>_q=0-<N>          (base debate)
#   target_c=1_de_r=<R>_q=0-<N>        (chosen,   actor)
#   target_c=0_de_r=<R>_q=0-<N>        (rejected, actor)
#   sup_t_c=1_de_r=<R>_q=0-<N> / sup_t_c=0_...   (critic, if GEN_SUPPORT=1)
#
# Tunables (env): NUMQ, ROUNDS, NUM_TRIALS, BATCH, GEN_SUPPORT
#
source "$( dirname "${BASH_SOURCE[0]}" )/config.sh"

NUMQ="${NUMQ:-9000}"
ROUNDS="${ROUNDS:-5}"
NUM_TRIALS="${NUM_TRIALS:-5}"
BATCH="${BATCH:-$NUMQ}"
GEN_SUPPORT="${GEN_SUPPORT:-0}"

COMMON=(
    --Qtype "$QTYPE"
    --num_questions "$NUMQ" --q_start 0 --num_rounds "$ROUNDS"
    --batch_size "$BATCH" --num_trials "$NUM_TRIALS"
    --roles none,detail
    --model_names "$MODEL,$MODEL"
    --support_list False,True
    --saved_model_list none,none
    --temps 0.7,0.7 --max_trys 2
    --use_judge False --resume False
    --bytenas "$STORAGE" --project_name "$PROJECT" --output_dir "$DATA_DIR"
    --hold_in_subjects None --hold_out_subjects None
)

echo "==================================================================="
echo " Stage 1: base debate trajectories -> $DATA_DIR"
echo "==================================================================="
$PYTHON debate_eval.py --debate_type debate --correct 1 "${COMMON[@]}"

echo "==================================================================="
echo " Stage 2: targeted (correct / incorrect) actor elicitation"
echo "==================================================================="
for C in 1 0; do
    echo "--- target correct=$C ---"
    $PYTHON debate_eval.py --debate_type target --correct "$C" \
        --debate_q_end "$NUMQ" "${COMMON[@]}"
done

if [ "$GEN_SUPPORT" = "1" ]; then
    echo "==============================================================="
    echo " Stage 2b: critic (support) targeted scoring"
    echo "==============================================================="
    for C in 1 0; do
        echo "--- sup_target correct=$C ---"
        $PYTHON debate_eval.py --debate_type sup_target --correct "$C" \
            --debate_q_end "$NUMQ" "${COMMON[@]}"
    done
fi

echo "Done. Generated files:"
ls -1 "$DATA_DIR"
