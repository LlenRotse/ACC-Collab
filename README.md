# LLM-Collab: training actor/critic LLMs to debate

---

## 1. Setup

Requires Python 3.10 and a CUDA (>= 12.1) GPU.

```bash
python3 -m venv ../venv                       # or a location of your choice
../venv/bin/pip install -r requirements.txt
```

The default models (Gemma-2, Llama, Mistral) are **gated** on HuggingFace, so
provide a token with access:

```bash
export HF_TOKEN=hf_xxx          # or: huggingface-cli login
```


---


## 3. Running each stage

All scripts read their configuration from `scripts/config.sh` and accept env
overrides. Run them from the repo root.

### 3.1 Data generation
```bash
QTYPE=BoolQ NUMQ=9000 NUM_TRIALS=5 bash scripts/run_data_generation.sh
```

How the data is built (the "goodness" signal used for filtering):
- **actor** pairs: quality = whether the elicited answer is correct; a pair is
  kept only when the chosen response beats the rejected one by ≥ threshold.
- **critic** pairs: quality = the actor's average accuracy on the *next* round
  after seeing the critic's response (a one-step improvement signal).

### 3.2 Training
```bash
# DPO on the actor (default)
bash scripts/run_dpo.sh
# DPO on the critic (needs GEN_SUPPORT=1 data)
SUPPORT=1 bash scripts/run_dpo.sh
# SFT variant
bash scripts/run_sft.sh
```
The merged model is written to `$STORAGE/dpo_out/$QTYPE/$MODEL/<tag>/LORA_DPO_<tag>`.

### 3.3 Inference / evaluation
```bash
# base model
QTYPE=BoolQ NUMQ=100 bash scripts/run_eval.sh
# trained actor
SAVED_MODEL=storage/dpo_out/BoolQ/gemma-2-2b-it/<tag>/LORA_DPO_<tag> \
    bash scripts/run_eval.sh
```
Prints per-round accuracy. `single_model_eval.py` evaluates a single model
without debate.


