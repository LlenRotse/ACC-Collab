from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers 

import numpy as np
import torch
import json
import yaml
import time
import os

import random

from utility import RegExHelper

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
import asyncio

import openai
import gc


if torch.__version__=='2.2.1+cu121' or transformers.__version__=='4.44.2':
    from vllm import LLM as VLLM
    from vllm import SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel


####################################
#### Local model cache (release) ###
####################################
# All models are downloaded from the HuggingFace Hub into this directory.
# Override with the LLM_COLLAB_CACHE environment variable if desired.
CACHE_DIR = os.environ.get("LLM_COLLAB_CACHE")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)

# vLLM tensor-parallel size for the local (V_*) models. Set LLM_COLLAB_TP>1 to
# shard a model across multiple GPUs (must match the visible GPU count).
TP_SIZE = int(os.environ.get("LLM_COLLAB_TP")





####################################
#### Huggingface authentication ####
####################################
# Gated models (Llama, Gemma, Mistral) require a HuggingFace token with access.
# Provide it via the HF_TOKEN environment variable or `huggingface-cli login`.
HF_TOKEN = os.environ.get("HF_TOKEN")
    


##################################
#### OpenAI with company key ####
####  all info is on lark    ####
##################################


# Only needed if you use GPT models (e.g. --model_names gpt-4). Set the
# OPENAI_API_KEY environment variable (and optionally OPENAI_BASE_URL for an
# OpenAI-compatible endpoint). The clients stay None when no key is configured
# so that importing this module never requires an OpenAI key.
def _make_openai_clients():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, None
    kwargs = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), AsyncOpenAI(**kwargs)


client, async_client = _make_openai_clients()
client_type = AsyncOpenAI






def get_data_type():

    device = torch.cuda.current_device()
    compute_capability = torch.cuda.get_device_capability(device)

    # bf16 is supported on compute capability 8.0 and higher
    if compute_capability >= (8, 0):
        data_type = torch.bfloat16
    else:
        data_type = torch.float16

    return data_type








class Embedding:

    def __init__(self, model_name="text-embedding-ada-002"):
        self.model = model_name
        self.reg = RegExHelper()

    def get_embeddings(self, responses):
        response_list = [self.reg.process(response) for response in responses]
        embs = client.embeddings.create(input=response_list, model=self.model).data
        return np.array([emb.embedding for emb in embs])
        # return client.embeddings.create(input = response_list, model=model).data[0].embedding
















class LLMHelper:

    def __init__(self, arnold=False, bytenas=''):

        self.reg = RegExHelper()
        self.max_trys = 10 # maximum number of tries to get a binary or fraction answer
        self.loaded_models = {}
        self.arnold = arnold
        self.bytenas = bytenas
        


    def load_llm(self, name, params=None, load_redundant_models=False, saved_model_path='None'):
        """
        Loads LLMs by name and handles loading redundant models (i.e., will pass a "pointer" to an already loaded
            model if an identical model is requested).
        :param name: name of the model
        :param params: params of the model (if none are passed, defaults are used)
        :param load_redundant_models: set to False in almost all cases
        :return: model, model_params
        """
        if name is None:
            return None, None

        gpt_names = ['gpt-4', 'gpt-3.5-turbo', 'text-davinci-003', 'gpt-4-1106-preview']
        t5_names = ['flan-t5-xxl', 'flan-t5-xl', 'flan-t5-large', 'flan-t5-base', 'flan-t5-small']
        alpaca_names = ['flan-alpaca-base', 'flan-alpaca-large', 'flan-alpaca-xl', 'flan-alpaca-xxl', 'flan-gpt4all-xl',
                        'flan-sharegpt-xl', 'flan-alpaca-gpt4-xl']
        Llama2_names = ['Llama-2-7b-chat-hf', 'Llama-2-7b-hf', 'Llama-2-13b-chat-hf', 'Llama-2-13b-hf']
        V_Llama2_names = ['V_Llama-2-7b-chat-hf', 'V_Llama-2-13b-chat-hf']
        V_Llama3_names = ['Meta-Llama-3-8B-Instruct']
        V_Mistral_names = ['Mistral-7B-Instruct-v0.2']
        V_Gemma_names  = ['Gemma-2-2b-it', 'gemma-2-2b-it']

        if load_redundant_models is False:
            if name in self.loaded_models:
                return self.loaded_models[name]

        if name in gpt_names:
            model = GPT()

        elif name in t5_names:
            model = FlanT5(arnold=self.arnold, bytenas=self.bytenas)

        elif name in alpaca_names:
            model = FlanAlpaca()
            
        elif name in Llama2_names:
            model = Llama2(arnold=self.arnold, bytenas=self.bytenas)
            
        elif name in V_Llama2_names or 'V_Llama-2-LORA' in name:
            model = V_Llama2(arnold=self.arnold, bytenas=self.bytenas, saved_model_path=saved_model_path)

        elif name in V_Llama3_names or 'V_Llama-3-LORA' in name:
            model = V_Llama3(arnold=self.arnold, bytenas=self.bytenas, saved_model_path=saved_model_path)

        elif name in V_Mistral_names or 'V_Mistral-LORA' in name:
            model = V_Mistral(arnold=self.arnold, bytenas=self.bytenas, saved_model_path=saved_model_path)
        elif name in V_Gemma_names or 'V_Gemma-2-LORA' in name:
            model = V_Gemma2(arnold=self.arnold, bytenas=self.bytenas, saved_model_path=saved_model_path)

        else:
            print(f'name=\"{name}\" is invalid in llms.py load_llm().')
            return None

        # default model params
        mld_params = {'temperature': 0.6,
                      'max_tokens': 256,
                      'model': name}

        if name not in gpt_names:
            mld_params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params is not None:
            for key in params:
                mld_params[key] = params[key]

        # keeps track of all models that have been loaded so far
        # that way we can recall repeat models
        self.loaded_models[name] = (model, mld_params)

        return model, mld_params











class LLM:
    # generic LLM class
    # all classes below are for specific LLMs
    def init(self):
        self.model = None


    ####################################################################
    ## The only two functions we care about are query and batch query ##
    ####################################################################
    def query(self, prompt, model, device, max_tokens=126, temperature=0.8):
        pass
    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8):
        pass





class V_Llama2(LLM):

    def __init__(self, arnold=False, bytenas='', saved_model_path='None'):

        self.valid_names = ['Llama-2-7b-chat-hf', 'Llama-2-7b-hf', 'Llama-2-13b-chat-hf', 'Llama-2-13b-hf']
        self.is_loaded = False

        self.model = None
        self.tokenizer = None

        self.bytenas = bytenas
        self.arnold = arnold

        self.saved_model_path = saved_model_path


    def load_llama2_model(self, name_or_patch):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
            bnb_4bit_use_double_quant=False,
        )

        if   '7b-hf' in name_or_patch:
            self.model_path = 'meta-llama/Llama-2-7b-hf'
            self.tokenizer_path = 'meta-llama/Llama-2-7b-hf'

        elif '13b-hf' in name_or_patch:
            self.model_path = 'meta-llama/Llama-2-13b-chat-hf'
            self.tokenizer_path = 'meta-llama/Llama-2-13b-chat-hf'

        elif '7b-chat' in name_or_patch:
            self.model_path = 'meta-llama/Llama-2-7b-chat-hf'
            self.tokenizer_path = 'meta-llama/Llama-2-7b-chat-hf'
        elif '13b-chat' in name_or_patch:
            self.model_path = 'meta-llama/Llama-2-13b-chat-hf'
            self.tokenizer_path = 'meta-llama/Llama-2-13b-chat-hf'

        elif 'LORA' in name_or_patch:
            # Load a fine-tuned (merged) checkpoint from a local directory,
            # with the base tokenizer.
            self.model_path = self.saved_model_path
            self.tokenizer_path = 'meta-llama/Llama-2-7b-chat-hf'
        else:
            print('NOT IMPLEMENTED')
            raise NotImplementedError


        dtype = get_data_type()

        self.model = VLLM(model=self.model_path, tokenizer=self.tokenizer_path,
                          download_dir=CACHE_DIR, gpu_memory_utilization=0.8,
                          tensor_parallel_size=TP_SIZE,
                          dtype=dtype, seed=random.randint(1, 100001))

        self.is_loaded = True

    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.6, logprobs=None, deload=False):


        sampling_params = SamplingParams(temperature=temperature, 
                                         max_tokens=max_tokens, 
                                         repetition_penalty=1, 
                                         min_tokens=10, 
                                         top_p=0.9, 
                                         logprobs=logprobs,
                                         seed=random.randint(1, 100001))

        if not self.is_loaded:
            self.load_llama2_model(model)


        outputs = self.model.generate(['[INST] ' + prompt + ' [/INST]' for prompt in prompt_batch], 
                                       sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]
        if deload:
            self.deload_model()

        outputs =None

        return outputs, responses

    def deload_model(self):
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.is_loaded = False



class V_Llama3(LLM):

    def __init__(self, arnold=False, bytenas='', saved_model_path='None'):

        self.valid_names = ['Meta-Llama-3-8B', 'Meta-Llama-3-8B-Instruct']
        self.is_loaded = False

        self.model = None
        self.tokenizer = None

        self.bytenas = bytenas
        self.arnold = arnold

        self.saved_model_path = saved_model_path

    def load_llama3_model(self, name_or_patch):
        
        if '8B' in name_or_patch:
            self.model_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
            self.tokenizer_path = 'meta-llama/Meta-Llama-3-8B-Instruct'

        elif 'LORA' in name_or_patch:
            # Load a fine-tuned (merged) checkpoint from a local directory,
            # with the base tokenizer.
            self.model_path = self.saved_model_path
            self.tokenizer_path = 'meta-llama/Meta-Llama-3-8B-Instruct'

        else:
            print('NOT IMPLEMENTED')
            raise NotImplementedError


        # Set the correct data type, is either bf16 or f16
        dtype = get_data_type()

        self.model = VLLM(model=self.model_path, tokenizer=self.tokenizer_path,
                          download_dir=CACHE_DIR, gpu_memory_utilization=0.8,
                          tensor_parallel_size=TP_SIZE,
                          dtype=dtype, seed=random.randint(1, 100001))

        self.helping_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True, cache_dir=CACHE_DIR)

        self.is_loaded = True


    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8, logprobs=None, deload=False):

        if not self.is_loaded:
            self.load_llama3_model(model)

        terminators = [
            self.helping_tokenizer.eos_token_id,
            self.helping_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        sampling_params = SamplingParams(temperature=temperature, 
                                         max_tokens=max_tokens, 
                                         repetition_penalty=1, 
                                         min_tokens=10, 
                                         top_p=0.9, 
                                         logprobs=logprobs,
                                         stop_token_ids=terminators,
                                         seed=random.randint(1, 100001))
        

        all_messages = [[{"role": "user", "content": prompt}] for prompt in prompt_batch
                ]

        formatted_prompt_batch = [self.helping_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in all_messages]

        outputs = self.model.generate(formatted_prompt_batch, 
                                       sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]

        if deload:
            self.deload_model()

        outputs = None

        return outputs, responses


    def deload_model(self):
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.is_loaded = False


class V_Mistral(LLM):

    def __init__(self, arnold=False, bytenas='', saved_model_path='None'):

        self.valid_names = ['Mistral-7B-Instruct-v0.2']
        self.is_loaded = False

        self.model = None
        self.tokenizer = None

        self.bytenas = bytenas
        self.arnold = arnold

        self.saved_model_path = saved_model_path

    def load_mistral_model(self, name_or_patch):
        if '7B' in name_or_patch:
            self.model_path = 'mistralai/Mistral-7B-Instruct-v0.2'
            self.tokenizer_path = 'mistralai/Mistral-7B-Instruct-v0.2'

        elif 'LORA' in name_or_patch:
            # Load a fine-tuned (merged) checkpoint from a local directory,
            # with the base tokenizer.
            self.model_path = self.saved_model_path
            self.tokenizer_path = 'mistralai/Mistral-7B-Instruct-v0.2'

        else:
            print('NOT IMPLEMENTED')
            raise NotImplementedError

        # Set the correct data type, is either bf16 or f16
        dtype = get_data_type()

        self.model = VLLM(model=self.model_path, tokenizer=self.tokenizer_path,
                          download_dir=CACHE_DIR, gpu_memory_utilization=0.8,
                          tensor_parallel_size=TP_SIZE,
                          dtype=dtype, seed=random.randint(1, 100001))

        self.is_loaded = True

    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8, logprobs=None, deload=False):

        sampling_params = SamplingParams(temperature=temperature, 
                                         max_tokens=max_tokens, 
                                         repetition_penalty=1, 
                                         min_tokens=10, 
                                         top_p=0.9, 
                                         logprobs=logprobs,
                                         seed=random.randint(1, 100001))

        if not self.is_loaded:
            self.load_mistral_model(model)

        outputs = self.model.generate(['[INST] ' + prompt + ' [/INST]' for prompt in prompt_batch], sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]

        if deload:
            self.deload_model()

        outputs = None

        return outputs, responses


    def deload_model(self):
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.is_loaded = False






class V_Gemma2(LLM):

    def __init__(self, arnold=False, bytenas='', saved_model_path='None'):

        self.valid_names = ['Gemma-2-2b-it', 'gemma-2-2b-it']
        self.is_loaded = False

        self.model = None
        self.tokenizer = None

        self.bytenas = bytenas
        self.arnold = arnold

        self.saved_model_path = saved_model_path

    def load_gemma2_model(self, name_or_patch):
        
        if 'gemma-2-2b-it' in name_or_patch:
            self.model_path     = 'google/gemma-2-2b-it'
            self.tokenizer_path = 'google/gemma-2-2b-it'
        elif 'LORA' in name_or_patch:
            # Load a fine-tuned (merged) checkpoint from a local directory,
            # with the base tokenizer.
            self.model_path     = self.saved_model_path
            self.tokenizer_path = 'google/gemma-2-2b-it'
        else:
            print('NOT IMPLEMENTED')
            raise NotImplementedError


        # Set the correct data type, is either bf16 or f16
        dtype = get_data_type()

        self.model = VLLM(model=self.model_path, tokenizer=self.tokenizer_path,
                          download_dir=CACHE_DIR, gpu_memory_utilization=0.8,
                          tensor_parallel_size=TP_SIZE,
                          dtype=dtype, seed=random.randint(1, 100001))

        self.helping_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True, cache_dir=CACHE_DIR)

        self.is_loaded = True


    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8, logprobs=None, deload=False):

        if not self.is_loaded:
            self.load_gemma2_model(model)

        sampling_params = SamplingParams(temperature=temperature, 
                                         max_tokens=max_tokens, 
                                         repetition_penalty=1, 
                                         min_tokens=10, 
                                         top_p=0.9, 
                                         logprobs=logprobs,
                                         seed=random.randint(1, 100001))
        

        all_messages = [[{"role": "user", "content": prompt}] for prompt in prompt_batch
                ]

        formatted_prompt_batch = [self.helping_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in all_messages]

        outputs = self.model.generate(formatted_prompt_batch, 
                                       sampling_params)
        
        responses = [output.outputs[0].text for output in outputs]

        if deload:
            self.deload_model()

        outputs = None

        return outputs, responses


    def deload_model(self):
        destroy_model_parallel()
        del self.model.llm_engine.model_executor.driver_worker
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.is_loaded = False






class Gemma2(LLM):

    def __init__(self, arnold=False, bytenas='', saved_model_path='None'):

        self.valid_names = ['Gemma-2-2b-it', 'gemma-2-2b-it']
        self.is_loaded = False

        self.model = None
        self.tokenizer = None

        self.bytenas = bytenas
        self.arnold = arnold

        self.saved_model_path = saved_model_path
        
        self.mini_batch_size=5

    def load_gemma2_model(self, name_or_patch):
        
        if 'gemma-2-2b-it' in name_or_patch:
            self.model_path     = 'google/gemma-2-2b-it'
            self.tokenizer_path = 'google/gemma-2-2b-it'
        elif 'LORA' in name_or_patch:
            self.model_path     = self.saved_model_path
            self.tokenizer_path = 'google/gemma-2-2b-it'
        else:
            print('NOT IMPLEMENTED')
            raise NotImplementedError
        
        # dtype = get_data_type()
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_compute_dtype='bfloat16',
        #     bnb_4bit_use_double_quant=False,
        # )


        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                            device_map="auto",
                                                            use_auth_token=HF_TOKEN,
                                                            # quantization_config=bnb_config,
                                                            )
        self.model = self.model.bfloat16()
        # self.model.bfloat16()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.helping_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.is_loaded = True


    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8, logprobs=None, deload=False):

        try:
            if not self.is_loaded:
                self.load_gemma2_model(model)

            
            ii = 0
            all_responses = []
            print()
            while ii < len(prompt_batch):
                mini_batch = prompt_batch[ii:ii+self.mini_batch_size]
                ii += self.mini_batch_size

                all_messages = [[{"role": "user", "content": prompt}] for prompt in mini_batch]
                formatted_prompt_batch = [self.helping_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in all_messages]

                inputs = self.tokenizer([prompt for prompt in formatted_prompt_batch],
                                        return_tensors="pt",
                                        padding="longest",
                                        )

                outputs = self.model.generate(input_ids=inputs['input_ids'].to(device),
                                            attention_mask=inputs['attention_mask'].to(device),
                                            max_new_tokens=max_tokens,
                                            do_sample=True,
                                            temperature=temperature,
                                            )
                
                responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                new_responses = []
                for resp in responses:
                    resp_to_use = resp.split('<start_of_turn>model')[1]
                    if '<end_of_turn>' in resp_to_use:
                        resp_to_use = resp_to_use.split('<end_of_turn>')[0]
                    new_responses.append(resp_to_use)
                responses = new_responses
                all_responses += responses
                print(f'Proccessing Gemma-2 batch: {round(ii/max(1, len(prompt_batch)), 4)}%,   {ii} out of {len(prompt_batch)}                         ', end='\r')
                outputs=None
                gc.collect()
                torch.cuda.empty_cache()
            print()
            if deload:
                self.deload_model()
        
            outputs = None

            return outputs, all_responses
        except:
            self.deload_model()
            print('#'*100)
            print('#'*100)
            print("MODEL OOM")
            print('#'*100)
            print('#'*100)
            return self.batch_query(prompt_batch, model, device, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs, deload=deload)


    def deload_model(self):
        # destroy_model_parallel()
        # del self.model.llm_engine.model_executor.driver_worker
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.is_loaded = False














class Gemma2(LLM):

    def __init__(self, arnold=False, bytenas='', saved_model_path='None'):

        self.valid_names = ['Gemma-2-2b-it', 'gemma-2-2b-it']
        self.is_loaded = False

        self.model = None
        self.tokenizer = None

        self.bytenas = bytenas
        self.arnold = arnold

        self.saved_model_path = saved_model_path
        
        self.mini_batch_size=5

    def load_gemma2_model(self, name_or_patch):
        
        if 'gemma-2-2b-it' in name_or_patch:
            self.model_path     = 'google/gemma-2-2b-it'
            self.tokenizer_path = 'google/gemma-2-2b-it'
        elif 'LORA' in name_or_patch:
            self.model_path     = self.saved_model_path
            self.tokenizer_path = 'google/gemma-2-2b-it'
        else:
            print('NOT IMPLEMENTED')
            raise NotImplementedError
        
        # dtype = get_data_type()
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type='nf4',
        #     bnb_4bit_compute_dtype='bfloat16',
        #     bnb_4bit_use_double_quant=False,
        # )


        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                            device_map="auto",
                                                            use_auth_token=HF_TOKEN,
                                                            # quantization_config=bnb_config,
                                                            )
        self.model = self.model.bfloat16()
        # self.model.bfloat16()
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.helping_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
        self.is_loaded = True


    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8, logprobs=None, deload=False):

        try:
            if not self.is_loaded:
                self.load_gemma2_model(model)

            
            ii = 0
            all_responses = []
            print()
            while ii < len(prompt_batch):
                mini_batch = prompt_batch[ii:ii+self.mini_batch_size]
                ii += self.mini_batch_size

                all_messages = [[{"role": "user", "content": prompt}] for prompt in mini_batch]
                formatted_prompt_batch = [self.helping_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in all_messages]

                inputs = self.tokenizer([prompt for prompt in formatted_prompt_batch],
                                        return_tensors="pt",
                                        padding="longest",
                                        )

                outputs = self.model.generate(input_ids=inputs['input_ids'].to(device),
                                            attention_mask=inputs['attention_mask'].to(device),
                                            max_new_tokens=max_tokens,
                                            do_sample=True,
                                            temperature=temperature,
                                            )
                
                responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
                new_responses = []
                for resp in responses:
                    resp_to_use = resp.split('<start_of_turn>model')[1]
                    if '<end_of_turn>' in resp_to_use:
                        resp_to_use = resp_to_use.split('<end_of_turn>')[0]
                    new_responses.append(resp_to_use)
                responses = new_responses
                all_responses += responses
                print(f'Proccessing Gemma-2 batch: {round(ii/max(1, len(prompt_batch)), 4)}%,   {ii} out of {len(prompt_batch)}                         ', end='\r')
                outputs=None
                gc.collect()
                torch.cuda.empty_cache()
            print()
            if deload:
                self.deload_model()
        
            outputs = None

            return outputs, all_responses
        except:
            self.deload_model()
            print('#'*100)
            print('#'*100)
            print("MODEL OOM")
            print('#'*100)
            print('#'*100)
            return self.batch_query(prompt_batch, model, device, max_tokens=max_tokens, temperature=temperature, logprobs=logprobs, deload=deload)


    def deload_model(self):
        # destroy_model_parallel()
        # del self.model.llm_engine.model_executor.driver_worker
        del self.tokenizer
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.is_loaded = False















class GPT:

    def __init__(self):
        if client is None or async_client is None:
            raise RuntimeError(
                "GPT models require an OpenAI API key. Set the OPENAI_API_KEY "
                "environment variable to use gpt-* models.")
        self.cost = 0
        self.attempt = 0
        self.sleep_time = 2
        self.client = client
        self.async_client = async_client

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(60))
    def query(self, prompt, model, device=None, temperature=0.8, max_tokens=512, logprobs=0):

        self.attempt += 1
        print('------')
        print(f'    GPT attempt #{self.attempt}')
        print('------')
        response, outputs = None, None
        if model in ['text-davinci-003']:
            outputs = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
            )
            response = outputs.choices[0].text.strip()

        elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4', 'gpt-4-1106-preview', 'gpt-3.5-turbo-instruct',
                       'text-davinci-002', 'gpt-35-turbo-16k',]:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}"}]
            outputs = client.chat.completions.create(model=model,
                                                     messages=messages,
                                                     temperature=temperature,
                                                     max_tokens=max_tokens,
                                                     )

            response = outputs.choices[0].message.content
        else:
            print(f'model=\"{model}\" is not a valid GPT model')

        self.cost += self.gpt_usage(model, outputs)

        self.attempt = 0

        return outputs, response


    async def query_(self, prompt, model, device=None, temperature=0.8, max_tokens=512, logprobs=0):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"}]
        outputs = None

        outputs = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=10,
            # logprobs=logprobs
        )

        return outputs

    async def batch_query_(self, prompt_batch, model, device=None, temperature=0.8, max_tokens=512, logprobs=0):
        all_outputs = await asyncio.gather(
            *[self.query_(prompt, model, device, temperature, max_tokens) for prompt in prompt_batch],
            return_exceptions=True)

        outputs = all_outputs
        responses = []
        for output in all_outputs:
            # print(type(output))
            if type(output) == openai.types.chat.chat_completion.ChatCompletion:
                responses.append(output.choices[0].message.content)
            else:
                print(f'INVALID OUTPUT FOUND: {output}')
                responses.append('Invalid Output')
        return all_outputs, responses

    def backup_batch_query(self, prompt_batch, model, device=None, temperature=0.8, max_tokens=512, logprobs=0):
        outputs_and_responses = [
            self.query(prompt, model=model, device=device, temperature=temperature, max_tokens=max_tokens,
                       logprobs=logprobs) for prompt in prompt_batch]

        outputs = [output for output, response in outputs_and_responses]
        responses = [response for output, response in outputs_and_responses]
        return outputs, responses


    def batch_query(self, prompt_batch, model, device=None, temperature=0.8, max_tokens=512, logprobs=0,
                    invalid_idx=None, iterative=False):
        self.attempt += 1

        if invalid_idx is None and len(prompt_batch) == 1:
            output, response = self.query(prompt_batch[0], model, device=device, temperature=temperature,
                                          max_tokens=max_tokens, logprobs=logprobs)
            return [output], [response]

        if iterative:
            outputs, responses = self.backup_batch_query(prompt_batch, model, device, temperature, max_tokens,
                                                         logprobs)
            return outputs, responses

        idx_to_use = list(range(len(prompt_batch)))
        if invalid_idx is not None:
            idx_to_use = invalid_idx

        prompts_to_use = [prompt_batch[i] for i in idx_to_use]
        outputs, responses = asyncio.run(
            self.batch_query_(prompts_to_use, model, device, temperature, max_tokens, logprobs))

        if invalid_idx is None:
            self.responses = responses
            self.outputs = outputs

        else:
            for i, idx in enumerate(idx_to_use):
                self.responses[idx] = responses[i]
                self.outputs[idx] = outputs[i]


        new_invalid_idx = []
        if 'Invalid Output' in responses:
            for resp in responses:
                print('- ', resp)
                print()
            sleep_time = int(self.sleep_time ** self.attempt)
            num_failures = len([response for response in responses if 'Invalid Output' in response])
            print(f'invlid_response found: wait_time={sleep_time}, num_failures={num_failures}')
            for ss in range(sleep_time):
                print(f'         waiting for {ss}', end='\r')
                time.sleep(1)
            print()
            # self.async_client = client_type(**async_client_params)


            for ii, response in enumerate(self.responses):
                if 'Invalid Output' in response:
                    new_invalid_idx.append(ii)

            self.outputs, self.responses = self.batch_query(prompt_batch, model, device, temperature, max_tokens,
                                                            logprobs, invalid_idx=new_invalid_idx, iterative=iterative)
        else:
            self.attempt = 0
        return self.outputs, self.responses

    def gpt_usage(self, backend, response):
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        if backend == 'gpt-4':
            cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
        elif backend == 'gpt-3.5-turbo':
            cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
        else:
            cost = completion_tokens / 1000 * 0.02 + prompt_tokens / 1000 * 0.02
        return cost


class FlanT5(LLM):

    def __init__(self, arnold=False, bytenas=''):
        super().__init__()
        self.is_loaded = False
        # self.model = None
        # self.tokenizer = None
        self.bytenas = bytenas
        self.arnold = arnold
        self.model_path = 'google/flan-t5-small'
        self.tokenizer_path = 'google/flan-t5-small'



    def load_t5_model(self, name_or_path):

        self.model = T5ForConditionalGeneration.from_pretrained('google/' + name_or_path, device_map="auto", cache_dir=CACHE_DIR)
        self.tokenizer = T5Tokenizer.from_pretrained('google/' + name_or_path, device_map='auto', cache_dir=CACHE_DIR)
        print('    ** loaded from remote file **')

        self.is_loaded = True
        # self.tokenizer = tokenizer
        # self.model = model    


    def query(self, prompt, model, device, max_tokens=126, temperature=0.8):

        if not self.is_loaded:
            self.load_t5_model(model)
        if self.model is None or self.tokenizer is None:
            print('flan-t5 tokenizer or model is None in llms.py [FlanT5.query()]')

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        outputs = self.model.generate(input_ids,
                                      max_new_tokens=max_tokens,
                                      do_sample=True,
                                      temperature=temperature,
                                      output_scores=True,
                                      min_new_tokens=10,
                                      return_dict_in_generate=True,
                                      )

        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return outputs, response

    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8):

        if not self.is_loaded:
            self.load_t5_model(model)
        if self.model is None or self.tokenizer is None:
            print('flan-t5 tokenizer or model is None in llms.py [FlanT5.query()]')

        inputs = self.tokenizer(prompt_batch,
                                return_tensors="pt",
                                padding="longest",
                                )

        outputs = self.model.generate(input_ids=inputs['input_ids'].to(device),
                                      attention_mask=inputs['attention_mask'].to(device),
                                      max_new_tokens=max_tokens,
                                      do_sample=True,
                                      temperature=temperature,
                                      output_scores=True,
                                      min_new_tokens=10,
                                      return_dict_in_generate=True,
                                      repetition_penalty=1.5
                                      )
        response = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        print(response)
        return outputs, response


class FlanAlpaca(LLM):

    def __init__(self):
        self.is_loaded = False
        self.model = None
        self.tokenizer = None

    def load_flan_alpaca_model(self, name_or_path):
        tokenizer = AutoTokenizer.from_pretrained("declare-lab/" + name_or_path, device_map='auto', cache_dir=CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained("declare-lab/" + name_or_path, device_map='auto', cache_dir=CACHE_DIR)

        self.is_loaded = True
        self.tokenizer = tokenizer
        self.model = model

        return tokenizer, model

    def query(self, prompt, model, device, max_tokens=126, temperature=0.8):

        if not self.is_loaded:
            self.load_flan_alpaca_model(model)
        if self.model is None or self.tokenizer is None:
            print('flan-alpaca tokenizer or model is None in llms.py [FlanAlpaca.query()]')

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        outputs = self.model.generate(input_ids,
                                      max_new_tokens=max_tokens,
                                      do_sample=True,
                                      temperature=temperature,
                                      output_scores=True,
                                      min_new_tokens=10,
                                      return_dict_in_generate=True,
                                      )

        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return outputs, response

    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8):

        if not self.is_loaded:
            self.load_flan_alpaca_model(model)
        if self.model is None or self.tokenizer is None:
            print('flan-t5 tokenizer or model is None in llms.py [FlanT5.query()]')

        inputs = self.tokenizer(prompt_batch,
                                return_tensors="pt",
                                padding="longest",
                                )

        outputs = self.model.generate(input_ids=inputs['input_ids'].to(device),
                                      attention_mask=inputs['attention_mask'].to(device),
                                      max_new_tokens=max_tokens,
                                      do_sample=True,
                                      temperature=temperature,
                                      output_scores=True,
                                      min_new_tokens=10,
                                      return_dict_in_generate=True,
                                      )
        response = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return outputs, response


class Llama2(LLM):

    def __init__(self, arnold=False, bytenas='', bnb=True):
        self.valid_names = ['Llama-2-7b-chat-hf', 'Llama-2-7b-hf', 'Llama-2-13b-chat-hf', 'Llama-2-13b-hf']
        self.is_loaded = False

        self.model = None
        self.tokenizer = None

        self.bytenas = bytenas
        self.arnold = arnold

        self.bnb = bnb

        self.model_path = 'meta-llama/Llama-2-7b-chat-hf'
        self.tokenizer_path = 'meta-llama/Llama-2-7b-chat-hf'

    def load_llama2_model(self, name_or_patch):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype='float16',
            bnb_4bit_use_double_quant=False,
        )

        if '7b' in name_or_patch:
            self.model_path = 'meta-llama/Llama-2-7b-chat-hf'
            self.tokenizer_path = 'meta-llama/Llama-2-7b-chat-hf'
        elif '13b' in name_or_patch:
            self.model_path = 'meta-llama/Llama-2-13b-chat-hf'
            self.tokenizer_path = 'meta-llama/Llama-2-13b-chat-hf'
        else:
            print('NOT IMPLEMENTED')
            raise NotImplementedError

        if self.bnb:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                            device_map="auto",
                                                            use_auth_token=HF_TOKEN,
                                                            cache_dir=CACHE_DIR,
                                                            quantization_config=bnb_config
                                                            )
            self.model = self.model.bfloat16()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                            device_map="auto",
                                                            use_auth_token=HF_TOKEN,
                                                            cache_dir=CACHE_DIR,
                                                            )
            self.model = self.model.bfloat16()

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True, use_auth_token=HF_TOKEN, cache_dir=CACHE_DIR)
        self.model.eval()

        self.is_loaded = True


    

    def batch_query(self, prompt_batch, model, device, max_tokens=126, temperature=0.8):

        if not self.is_loaded:
            self.load_llama2_model(model)

        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(['[INST]' + prompt + '[/INST]' for prompt in prompt_batch],
                                return_tensors="pt",
                                padding="longest",
                                )

        outputs = self.model.generate(input_ids=inputs['input_ids'].to(device),
                                      attention_mask=inputs['attention_mask'].to(device),
                                      max_new_tokens=max_tokens,
                                      do_sample=True,
                                      temperature=temperature,
                                      output_scores=True,
                                      return_dict_in_generate=True,
                                      )
        responses = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        new_responses = []
        for response in responses:
            idx = response.find('[/INST]') + len('[/INST]')
            new_responses.append(response[idx:])

        responses = new_responses
        torch.cuda.empty_cache()
        return outputs, responses
