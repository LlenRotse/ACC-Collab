from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from transformers import TrainerCallback
from transformers import EarlyStoppingCallback
import os
from trl.commands.cli_utils import init_zero_verbose, SFTScriptArguments, TrlParser
import torch
from datasets import load_dataset

from tqdm.rich import tqdm
from transformers import AutoTokenizer, TrainingArguments

from trl import (
    ModelConfig,
    SFTTrainer,
    SFTConfig
)
import bitsandbytes as bnb

tqdm.pandas()

from dataclasses import dataclass, field
from typing import Optional

from data import DatasetCreater

# --- Local storage / auth (release) ---
# Models are downloaded from the HuggingFace Hub into CACHE_DIR. Gated models
# (Llama, Gemma, Mistral) need a token: set HF_TOKEN or `huggingface-cli login`.
CACHE_DIR = os.environ.get("LLM_COLLAB_CACHE", "/opt/tiger/AgentMonitor/tmp")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Weights & Biases is optional; disabled by default so training runs without a
# login. Set WANDB_MODE=online (and WANDB_API_KEY) to enable experiment logging.
os.environ.setdefault("WANDB_MODE", "disabled")

@dataclass
class AdditionalArguments:
    json_path: Optional[str] = field(default=None, metadata={"help": "Path to the dataset JSON file"})
    debate_files: Optional[str] = field(default=None, metadata={"help": "List of debate files to use for data."})
    lora_r_: Optional[int] = field(default=None, metadata={"help": "LORA R parameter"})
    support: Optional[int] = field(default=None, metadata={"help": "Indicates whether to train support, or non-support, models"})
    
    num_train_questions: int = field(default=0, metadata={"help": "Number of questions to use in training"})
    num_eval_questions: int = field(default=0, metadata={"help": "Number of questions to use in evaluation"})
    support_persona: str = field(default='helper', metadata={"help": "persona of the support that you are training"})
    cot: int = field(default=0, metadata={"help": "1 to do CoT, 0 to not do CoT"})
    rounds_to_use: str = field(default='', metadata={"help": "rounds to use during training"})

    max_train_examples: int = field(default=-1)
    max_eval_examples: int = field(default=-1)



class LogCallback(TrainerCallback):
        """A custom callback to log losses to a text file."""
        
        def __init__(self, log_file):
            self.log_file = log_file
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called every logging step."""
            if logs is not None:
                with open(self.log_file, "a") as log_fp:
                    log_fp.write(f"{state.global_step}\t{logs.get('loss', 'N/A')}\t{logs.get('eval_loss', 'N/A')}\n")





def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            # text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
            instruction = example['instruction'][i]
            output      = example['output'][i]
            text = f"[INST] {example['instruction'][i]} [/INST] \n{example['output'][i]}"
            output_texts.append(text)
        return output_texts


def id_format(example):
    return [example['instruction'][i] + '\n' + example['output'][i] for i in range(len(example['instruction']))]


def find_all_linear_names(model):
    """
    Find modules to apply LoRA to.
    :param model: PEFT model
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, TrainingArguments, ModelConfig, AdditionalArguments))
    args, training_args, model_config, add_args = parser.parse_args_and_config()
    
    lr = training_args.learning_rate
    lora_r_ = int(add_args.lora_r_)
    num_train_epochs = int(training_args.num_train_epochs)
    num_train_q = int(add_args.num_train_questions)
    num_eval_q = int(add_args.num_eval_questions)
    support = int(add_args.support)
    # max_seq_length = int(args.max_seq_length)
    max_seq_length = 1000

    if add_args.rounds_to_use == '':
        rounds_to_use = None
    else:
         rounds_to_use = [int(s) for s in add_args.rounds_to_use.split(',')]
    
    tag = f'lr{lr}_lora_r{lora_r_}_tr_ep{num_train_epochs}_tr_q{num_train_q}_eval_q{num_eval_q}_supp{support}_max_len{max_seq_length}'.replace('.', '_')      
    
    training_args.logging_first_step = False
    training_args.eval_on_start = False

    training_args.max_grad_norm = 0.3
    training_args.warmup_ratio = 0.03
    training_args.lr_scheduler_type='cosine'
    # training_args.weight_decay = 0.00
    # training_args.gradient_accumulation_steps=1

    training_args.output_dir = training_args.output_dir + f'/{tag}'
    log_file = os.path.join(training_args.output_dir + f'/logs/{tag}', 'logs.txt')
    os.makedirs(training_args.output_dir + f'/logs/{tag}' , exist_ok=True)
    
    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    
    
    # ################
    # # Dataset
    # ################
    # # Load the dataset
    # full_dataset = load_dataset("json", data_files=add_args.json_path, split="train")
    # # full_dataset = load_dataset(args.dataset_name)

    # # If the dataset does not come with predefined splits, split the dataset manually
    # train_test_split = full_dataset.train_test_split(test_size=0.1)
    # train_dataset = train_test_split['train']
    # eval_dataset = train_test_split['test']



    ####################
    ### Training params
    ####################
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
            bnb_4bit_use_double_quant=False,
        )

    if 'gemma-2' in model_config.model_name_or_path:
        base_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,
                                                        device_map="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        quantization_config=bnb_config,
                                                        token=HF_TOKEN,
                                                        cache_dir=CACHE_DIR,
                                                        attn_implementation='eager'
                                                        )
    else:
         base_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,
                                                        device_map="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        quantization_config=bnb_config,
                                                        token=HF_TOKEN,
                                                        cache_dir=CACHE_DIR,
                                                        )

    
    # Change the LORA hyperparameters accordingly to fit your use case
    peft_config = LoraConfig(r=add_args.lora_r_,
                            lora_alpha=add_args.lora_r_*2,
                            target_modules=find_all_linear_names(base_model),
                            lora_dropout=0.05,
                            bias="none",
                            task_type="CAUSAL_LM",)
    
    dataset_creater = DatasetCreater()
    file_name_list = [add_args.json_path + file_name for file_name in add_args.debate_files.split(',')]

    helping_tokenizer = None
    if 'Llama-3' in model_config.model_name_or_path or 'gemma-2' in model_config.model_name_or_path:
        helping_tokenizer = tokenizer
    dataset = dataset_creater.build_single_dataset(json_path=add_args.json_path,
                                                   pattern_list=add_args.debate_files.split(','),
                                                   num_train_questions=add_args.num_train_questions, 
                                                   num_eval_questions=add_args.num_eval_questions,
                                                   support=bool(add_args.support), 
                                                   max_train_examples=add_args.max_train_examples, 
                                                   max_eval_examples=add_args.max_eval_examples, 
                                                   prompt_formatting_function=None,
                                                   support_persona=add_args.support_persona, 
                                                   comparison=False, 
                                                   threshold=0.8,
                                                   tokenizer=helping_tokenizer,
                                                   rounds_to_use=rounds_to_use,
                                                    )
    train_dataset = dataset['train']
    eval_dataset  = dataset['eval']


    response_template = '[/INST]'
    if   'Llama-3' in model_config.model_name_or_path:
         response_template = '<|end_header_id|>'
    elif 'gemma-2' in model_config.model_name_or_path:
         response_template = '<start_of_turn>'
         
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    print("#####\n"*3)
    print(f'weight_decay          = {training_args.weight_decay}')
    print(f'gradient_accumulation = {training_args.gradient_accumulation_steps}')
    print("#####\n"*3)


    formatting_func = formatting_prompts_func
    if 'Llama-3' in model_config.model_name_or_path or 'gemma-2' in model_config.model_name_or_path:        
        formatting_func = id_format 

    trainer = SFTTrainer(model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
        data_collator=collator,
        callbacks=[LogCallback(log_file), EarlyStoppingCallback(early_stopping_patience=20)],
        
        )
        
    print(trainer.evaluate())
    trainer.train()
    adapter_model_name = training_args.output_dir + f'/adapt_sft_{tag}'
    trainer.save_model(adapter_model_name)

    ################
    # Merge and save model into a new folder
    ################
    # Create merged model folder
    # dataset_name = add_args.json_path.split('/')[-1].split('.')[0]
    # merged_model_name = training_args.output_dir + f'/LORA_SFT_{add_args.lora_r_}_{dataset_name}'
    merged_model_name = training_args.output_dir + f'/LORA_SFT_{tag}'

    print("#"*200)
    print("#"*200)
    print(f'    merged_model_name: {merged_model_name}')
    print("#"*200)
    print("#"*200)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.save_pretrained(merged_model_name)

    # Save merged model
    base_model_name = model_config.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base_model_name, token=HF_TOKEN, cache_dir=CACHE_DIR)
    model = PeftModel.from_pretrained(model, adapter_model_name)

    model = model.merge_and_unload()
    model.save_pretrained(merged_model_name)

    
    
    # Testing for evaluation and loading model
    

    # tokenizer = AutoTokenizer.from_pretrained(base_model_name)


    # inputs = tokenizer(["Today is"], return_tensors="pt")

    # # Example 1: Print the scores for each token generated with Greedy Search
    # outputs = model.generate(**inputs, max_new_tokens=512)

    # # The generated tokens are stored in `outputs.sequences`
    # generated_tokens = outputs

    # # The scores for each token are stored in `outputs.scores`
    # # Each entry in `outputs.scores` is a tensor representing the scores for each token in the vocabulary at that step

    # # Convert generated tokens to text
    # generated_text = tokenizer.decode(generated_tokens[0])
    # breakpoint()
