
import utility
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from transformers import EarlyStoppingCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from transformers import BitsAndBytesConfig
from peft import PeftModel, LoraConfig
import bitsandbytes as bnb

from datasets import Dataset, load_dataset

import torch
from trl import DPOTrainer, ModelConfig, DPOConfig

from trl.commands.cli_utils import TrlParser

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from transformers import TrainingArguments
from transformers import TrainerCallback, EarlyStoppingCallback

from data import DatasetCreater
import os

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
import wandb

class LogCallback(TrainerCallback):
        """A custom callback to log losses to a text file."""

        def __init__(self, log_file):
            self.log_file = log_file

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Called every logging step."""
            if logs is not None:
                with open(self.log_file, "a") as log_fp:
                    log_fp.write(f"{state.global_step}\t{logs.get('loss', 'N/A')}\t{logs.get('eval_loss', 'N/A')}\n")


# @dataclass
# class DPOConfig(TrainingArguments):
#     beta: float = 0.1
#     label_smoothing: float = 0
#     loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair"] = "sigmoid"
#     label_pad_token_id: int = -100
#     padding_value: int = 0
#     truncation_mode: str = "keep_end"
#     max_length: Optional[int] = None
#     max_prompt_length: Optional[int] = None
#     max_target_length: Optional[int] = None
#     is_encoder_decoder: Optional[bool] = None
#     disable_dropout: bool = True
#     generate_during_eval: bool = False
#     precompute_ref_log_probs: bool = False
#     dataset_num_proc: Optional[int] = None
#     model_init_kwargs: Optional[Dict] = None #{'max_tokens': 512, 'temperature': 0.6}
#     ref_model_init_kwargs: Optional[Dict] = None
#     model_adapter_name: Optional[str] = None
#     ref_adapter_name: Optional[str] = None
#     reference_free: bool = False
#     force_use_ref_model: bool = False
#     per_device_train_batch_size: int = 10
#     per_device_eval_batch_size: int = 10
#     auto_find_batch_size: bool =True
#     output_dir: str ='/opt/tiger/LLM-Collab/dpo_outputs'
#     beta: float = 0.1
#     num_train_epochs: int = 10
#     max_steps:int = -1
#     evaluation_strategy:str = 'steps'
#     learning_rate:float = 1.41e-5
#     save_steps: int = 200
#     max_grad_norm: float = 0.3
#     warmup_ratio: float = 0.03
#     lr_scheduler_type='cosine'




@dataclass
class AdditionalArguments:
    json_path: Optional[str] = field(default=None, metadata={"help": "Path to the dataset JSON file"})
    debate_files: Optional[str] = field(default=None, metadata={"help": "List of debate files to use for data."})
    lora_r_: Optional[int] = field(default=None, metadata={"help": "LORA R parameter"})
    support: Optional[int] = field(default=None, metadata={"help": "Indicates whether to train support, or non-support, models"})
    
    num_train_questions: int = field(default=0, metadata={"help": "Number of questions to use in training"})
    num_eval_questions: int = field(default=0, metadata={"help": "Number of questions to use in evaluation"})
    support_persona: str = field(default='helper', metadata={"help": "persona of the support that you are training"})
    lr: float = field(default=1.41e-5, metadata={"help": "Learning rate ffor DPO training"})
    max_eval_examples:  int = field(default=-1)
    max_train_examples: int = field(default=-1)
    
    support_persona: str = field(default='helper', metadata={"help": "persona of the support that you are training"})
    cot: int = field(default=0, metadata={"help": "1 to do CoT, 0 to not do CoT"})
    project_name: str = field(default='DEBUG_WANDB', metadata={"help": "wandb project name"}) 
    hold_out_subjects: str = field(default='None', metadata={"help": "For MMLU, subjects to keep out of the dataset"}) 
    hold_in_subjects: str = field(default='None', metadata={"help": "For MMLU, subjects to keep in the dataset"}) 



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


def prompt_formatting_function(prompt):
    return f'[INST] {prompt} [/INST]'

def id_format(prompt):
    return prompt




if __name__=='__main__':
    
    parser = TrlParser((DPOConfig, ModelConfig, AdditionalArguments))
    training_args, model_config, add_args = parser.parse_args_and_config()
    
    if add_args.hold_in_subjects == 'None' or add_args.hold_in_subjects == '':
        add_args.hold_in_subjects = None
    else:
        add_args.hold_in_subjects = add_args.hold_in_subjects.split(',')
        
    if add_args.hold_out_subjects == 'None' or add_args.hold_out_subjects == '':
        add_args.hold_out_subjects = None
    else:
        add_args.hold_out_subjects = add_args.hold_out_subjects.split(',')
    
    lr = training_args.learning_rate
    lora_r_ = int(add_args.lora_r_)
    num_train_epochs = int(training_args.num_train_epochs)
    num_train_q = int(add_args.num_train_questions)
    num_eval_q = int(add_args.num_eval_questions)
    support = int(add_args.support)
    
    max_prompt_length = int(training_args.max_prompt_length)
    max_target_length = int(training_args.max_target_length)
    
    if training_args.max_length:
        raise ValueError("max_length should not be set in the training arguments. We use max_prompt_length + max_target_length instead.")
    
    max_length = max_prompt_length + max_target_length
    rpo_alpha = training_args.rpo_alpha


    training_args.logging_first_step = False
    training_args.eval_on_start = False
    
    tag = f'rpo_alpha{rpo_alpha}_lr{lr}_lora_r{lora_r_}_tr_ep{num_train_epochs}_tr_q{num_train_q}_eval_q{num_eval_q}_supp{support}_max_len{max_length}'.replace('.', '_')   
    training_args.output_dir = training_args.output_dir + f'/{tag}'
    log_file = os.path.join(training_args.output_dir+f'/logs/{tag}', 'logs.txt')
    os.makedirs(training_args.output_dir+f'/logs/{tag}', exist_ok=True)
    
    wandb.init(project=add_args.project_name, name= str(training_args.rpo_alpha) + '_' + tag, config=training_args)

    
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
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,
                                                        device_map="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        quantization_config=bnb_config,
                                                        token=HF_TOKEN,
                                                        cache_dir=CACHE_DIR,
                                                        attn_implementation='eager'
                                                        )
    else:
         model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,
                                                        device_map="auto",
                                                        torch_dtype=torch.bfloat16,
                                                        quantization_config=bnb_config,
                                                        token=HF_TOKEN,
                                                        cache_dir=CACHE_DIR,
                                                        )

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(r=add_args.lora_r_,
                            lora_alpha=add_args.lora_r_*2,
                            target_modules=find_all_linear_names(model),
                            lora_dropout=0, # DPO optimized for dropout = 0 (different from SFT)
                            bias="none",
                            task_type="CAUSAL_LM")


    file_name_list = [add_args.json_path + file_name for file_name in add_args.debate_files.split(',')]
    

    helping_tokenizer = None
    formatting_function = prompt_formatting_function
    if 'Llama-3' in model_config.model_name_or_path or 'gemma-2' in model_config.model_name_or_path:
        helping_tokenizer = tokenizer
        formatting_function = id_format



    dataset_creater = DatasetCreater()
    if add_args.cot == 1:
       print('#\n'*5, "COT is depricated", '\n#'*5)
       exit()

    else:
        dataset = dataset_creater.build_preference_dataset(json_path=add_args.json_path,
                                                           pattern_list=add_args.debate_files.split(','),
                                                           num_train_questions=add_args.num_train_questions, 
                                                           num_eval_questions=add_args.num_eval_questions,
                                                           support=bool(add_args.support), 
                                                           max_train_examples=add_args.max_train_examples, 
                                                           max_eval_examples=add_args.max_eval_examples, 
                                                           prompt_formatting_function=formatting_function,
                                                           support_persona=add_args.support_persona, 
                                                           comparison=True, 
                                                           threshold=0.6,
                                                           hold_in_subjects=add_args.hold_in_subjects,
                                                           hold_out_subjects=add_args.hold_out_subjects,
                                                           tokenizer=helping_tokenizer
                                                           )

    train_dataset = dataset['train']
    eval_dataset  = dataset['eval']

    print('Dataset: ', dataset,       end='\n\n\n')
    print('Trainset:', train_dataset, end='\n\n\n')
    print('Evalset: ', eval_dataset,  end='\n\n\n')



    print("#####\n"*3)
    print(f'weight_decay          = {training_args.weight_decay}')
    print(f'gradient_accumulation = {training_args.gradient_accumulation_steps}')
    print("#####\n"*3)

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['eval'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        max_length=max_length,
        callbacks=[LogCallback(log_file), EarlyStoppingCallback(early_stopping_patience=5)]
    )
    print(dpo_trainer.evaluate())
    dpo_trainer.train()
    adapter_model_name = training_args.output_dir + f'/adapt_dpo_{tag}'
    dpo_trainer.save_model(adapter_model_name)

    ################
    # Merge and save model into a new folder
    ################
    # # Create merged model folder
    # dataset_name = add_args.json_path.split('/')[-1].split('.')[0]
    # merged_model_name = training_args.output_dir + f'/LORA_DPO_{add_args.lora_r_}_{dataset_name}'
    merged_model_name = training_args.output_dir + f'/LORA_DPO_{tag}'
    
    print("Merged model file path:",  merged_model_name)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True, token=HF_TOKEN, cache_dir=CACHE_DIR)
    tokenizer.save_pretrained(merged_model_name)

    # Save merged model
    base_model_name = model_config.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base_model_name, token=HF_TOKEN, cache_dir=CACHE_DIR)
    model = PeftModel.from_pretrained(model, adapter_model_name)

    model = model.merge_and_unload()
    model.save_pretrained(merged_model_name)

    

