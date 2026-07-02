import os
import utility
import numpy as np
from datasets import load_dataset
import json
import pandas as pd
import random

from questions import BoolQuestion

# Directory for the intermediate temp_{train,eval}_data.{csv,json} files. Defaults
# to the current directory; set LLM_COLLAB_TMP to a unique path per run so that
# several training jobs can build their datasets concurrently without clobbering.
TMP_DIR = os.environ.get("LLM_COLLAB_TMP", ".")
os.makedirs(TMP_DIR, exist_ok=True)



def combine_dicts(json_path, pattern, num_questions_per_file, q_start, q_end, key_type='int_tuple', verbose=False):
    
    combined_d = {}
    for file in os.listdir(json_path):

        # grab all files with the given pattern
        if pattern in file:
            file_q_start = int(file.split('_q=')[1].split('-')[0])
            file_q_end   = int(file.split('_q=')[1].split('-')[1])
            
            # file does not contain the correct number of questions
            if file_q_end-file_q_start != num_questions_per_file:
                continue
                
            # file contains no questions between q_start and q_end
            if file_q_start > q_end or file_q_end < q_start:
                continue
            
            print(file)
            loaded_d = utility.load_dict(json_path+file, key_type=key_type, verbose=verbose)
            print(len(loaded_d))
            
            
            for key in loaded_d:
                # offset the key of loaded_d
                if key_type == 'int_tuple':
                    new_key = (key[0]+file_q_start, key[1])
                    q_id = new_key[0]
                elif key_type == 'int':
                    new_key = key[0]+file_q_start
                    q_id = new_key
                else:
                    print(f'key_type={key_type} is not valid, must be "int" or "int_tuple"')
                    exit(-1)
                # check if the current question is outside (q_start, q_end)
                if q_id < q_start or q_end < q_id:
                    continue
                else:
                    combined_d[new_key] = loaded_d[key]
                
    return combined_d





class DatasetCreater:


    def __init__(self):
        pass
    

    def build_preference_dataset(self, json_path, pattern_list, num_train_questions, num_eval_questions, support=True, max_train_examples=None, max_eval_examples=None, prompt_formatting_function=None, 
                                 support_persona=None, comparison=True, threshold=0.5, hold_out_subjects=None, hold_in_subjects=None, round_balance=True, tokenizer=None, rounds_to_use=None):

        if hold_out_subjects is not None and hold_in_subjects is not None:
            print(f"either hold_out_subjects and hold_in_subjects must be None, but you have\nhold_out_subjects={hold_out_subjects}\nhold_in_subjects={hold_in_subjects}")
            exit(-1)
        file_name_list = []
        for file in os.listdir(json_path):
            if any(pattern in file for pattern in pattern_list):
                file_name_list.append(json_path+file)
        
        if support_persona not in ['helper', 'none', 'detail', 'ask-question', 'judge', 'test-persona']:
            print("#"*500, '\n', "#"*500)
            print(f"##### DPO currently only supports helper and none, but you have support_persona={support_persona} ####")
            print("#"*500, '\n', "#"*500)
            exit()

        train_dict = {'prompt': [], 'chosen': [], 'rejected': []}
        eval_dict  = {'prompt': [], 'chosen': [], 'rejected': []}

        # question_set = BoolQuestion()
        if len(file_name_list)==1:
            file_name_list = file_name_list*2
        all_examples = {}
        correct = -1
        random.seed(101)
        suffled_idx = list(range(num_train_questions + num_eval_questions))
        random.shuffle(suffled_idx)

        for iii, file_name in enumerate(file_name_list):
            print('loading file:', file_name)
            
            # if 'c=0' not in file_name and 'c=1' not in file_name:
            #     print(f'file_name={file_name} is not valid for DPO data, must have c=0 or c=1 in the file name (this specifies target)')
            #     continue
            
            
            d = utility.load_dict(file_name, key_type='int_tuple', verbose=True)

            # dataset_lims = {'BoolQ': 9000, 'MMLUQ': 10_000, 'SCIQ': 11_000, 'BBH': 4_000, 'MedMCQA': 10_000, 'ARC': 4_000}
            # Qtype = None
            # for dataset_name in dataset_lims:
            #     if dataset_name in json_path:
            #         Qtype = dataset_name
            # d = {(q, t): d[(q, t)] for q, t in d if q < dataset_lims[Qtype]}

            keys = np.array(list(d.keys()))
            max_t = len(set(keys[:,1]))
            new_d = {}
            for q in range(len(suffled_idx)):
                for t in range(max_t):
                    new_d[(q, t)] = d[(suffled_idx[q], t)]
            d = new_d
            
            keys = np.array(list(d.keys()))
            
            question_ids = sorted(list(set(keys[:,0])))
            trail_ids    = sorted(list(set(keys[:,1])))

            q_start = int(file_name.split('q')[-1].split('-')[0].replace('=',''))
            if all('_c=' in file_ for file_ in file_name_list):
                correct = int(file_name.split('_c=')[1].split('_')[0]) # whether target is correct (c=1) or incorrect (c=1)
            elif all('_c=' not in file_ for file_ in file_name_list):
                correct = iii
            else:
                print('Cannot have a mix of c=X files and files without c=x, double check your debate_files')
                exit()

            rounds  = int(file_name.split('_r=')[1].split('_')[0]) # number of rounds
            mld_idx = 1 if support else 0  # whether to grab data from the first or second model

            for q in question_ids:
                for t in trail_ids:
                    if (q, t) not in d or q + q_start > num_train_questions + num_eval_questions:
                        continue                  
                    # filtering out hold-out-subjects
                    if   hold_out_subjects is not None and d[(q, t)]['subject_type'] in hold_out_subjects:
                        continue
                    # filtering anything not in hold-in-subjects
                    elif hold_in_subjects is not None and d[(q, t)]['subject_type'] not in hold_in_subjects:
                        continue

                    if support:
                        r_start = 0
                    elif not support:
                        r_start = 1
                    if rounds_to_use is None:
                        rounds_to_use = range(r_start, rounds)
                    for r in rounds_to_use:
                        if (q+q_start, t, r) not in all_examples:
                            all_examples[(q+q_start, t, r)] = [{}, {}]

                        prompt   = d[(q, t)]['all_prompts'][r][mld_idx]
                        response = d[(q, t)]['all_resps'][r][mld_idx]

                        # quality of support data is computed via average accuracy of the next round (rather than one-shot accuray of the current round)
                        if support:
                            quality  = d[(q, t)]['avg_accs'][r][0]
                        else:
                            quality  = d[(q, t)]['all_preds'][r][mld_idx] == d[(q, t)]['answer']
                        
                        all_examples[(q+q_start, t, r)][correct] = {'prompt': prompt, 'response': response, 'quality': quality, 'round': r}
        
        mean_val = []
        train_rounds = []
        eval_rounds  = []
        for key in all_examples:
            neg = all_examples[key][0]
            pos = all_examples[key][1]
            
            assert neg['prompt'] == pos['prompt']
            prompt = pos['prompt']
            
            if prompt_formatting_function is not None:
                prompt = prompt_formatting_function(prompt)
            
            if tokenizer is not None:
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            
            val = pos['quality'] - neg['quality']
            mean_val.append(abs(val))
            if not comparison:
                val = pos['quality']

            if val >= threshold:
                if key[0] <= num_train_questions:    
                    train_dict['prompt'].append(prompt)
                    train_dict['chosen'].append(pos['response'])
                    train_dict['rejected'].append(neg['response'])
                    train_rounds.append(pos['round'])
                elif num_train_questions < key[0] <= num_train_questions + num_eval_questions:
                    eval_dict['prompt'].append(prompt)
                    eval_dict['chosen'].append(pos['response'])
                    eval_dict['rejected'].append(neg['response'])
                    eval_rounds.append(pos['round'])
        
        if max_train_examples is not None:

            train_idx = list(range(len(train_dict['prompt'])))
            
            # train_rounds = np.array(train_rounds)
            # train_weight = [1/np.sum(train_rounds==r) for r in range(rounds)]
            # train_idx = np.random.choice(train_idx, size=max_train_examples, replace=False, p=train_weight)
            max_train_examples = min(max_train_examples, len(train_idx))
            train_idx = np.random.choice(train_idx, size=max_train_examples, replace=False)
            # random.shuffle(train_idx)

            for key in eval_dict.keys():
                train_dict[key] = [train_dict[key][ii] for ii in train_idx]

            # train_dict['prompt']   = train_dict['prompt'][:max_train_examples]
            # train_dict['chosen']   = train_dict['chosen'][:max_train_examples]
            # train_dict['rejected'] = train_dict['rejected'][:max_train_examples]


        if max_eval_examples is not None:
            eval_idx = list(range(len(eval_dict['prompt'])))
            
            # eval_rounds = np.array(eval_rounds)
            # eval_weight = [1/np.sum(eval_rounds==r) for r in range(rounds)]
            # eval_idx = np.random.choice(eval_idx, size=max_eval_examples, replace=False, p=eval_weight)

            max_eval_examples = min(max_eval_examples, len(eval_idx))
            eval_idx = np.random.choice(eval_idx, size=max_eval_examples, replace=False)

            for key in eval_dict.keys():
                eval_dict[key]  = [eval_dict[key][ii]  for ii in eval_idx]

            # eval_dict['prompt']   = eval_dict['prompt'][:max_eval_examples]
            # eval_dict['chosen']   = eval_dict['chosen'][:max_eval_examples]
            # eval_dict['rejected'] = eval_dict['rejected'][:max_eval_examples]


        print()
        print()
        print('Len Train:', len(train_dict['prompt']))
        print('Len Eval: ', len(eval_dict['prompt']))
        print()
        print()
        print(np.mean(mean_val))
        return self.dicts_to_dpo_dataset(train_dict, eval_dict)



    def build_single_dataset(self, json_path, pattern_list, num_train_questions, num_eval_questions, support=True, max_train_examples=None, max_eval_examples=None, prompt_formatting_function=None, support_persona=None, comparison=False, threshold=0.5, hold_out_subjects=None, tokenizer=None, rounds_to_use=None):
        dataset = self.build_preference_dataset(json_path, 
                                                pattern_list,
                                                num_train_questions, 
                                                num_eval_questions, 
                                                support=support, 
                                                max_train_examples=max_train_examples, 
                                                max_eval_examples=max_eval_examples, 
                                                prompt_formatting_function=prompt_formatting_function, 
                                                support_persona=support_persona, 
                                                comparison=comparison, 
                                                threshold=threshold,
                                                hold_out_subjects=hold_out_subjects,
                                                tokenizer=tokenizer,
                                                rounds_to_use=rounds_to_use,
                                                )
        
        train_dataset = dataset['train']
        eval_dataset  = dataset['eval']
        
        train_data = []
        for prompt, chosen, rejected in zip(train_dataset['prompt'], train_dataset['chosen'], train_dataset['rejected']):
            train_data.append({'instruction': prompt, 'output': chosen})
            
        eval_data = []
        for prompt, chosen, rejected in zip(eval_dataset['prompt'], eval_dataset['chosen'], eval_dataset['rejected']):
            eval_data.append({'instruction': prompt, 'output': chosen})
        

        train_json = os.path.join(TMP_DIR, 'temp_train_data.json')
        eval_json  = os.path.join(TMP_DIR, 'temp_eval_data.json')
        with open(train_json, 'w', encoding='utf8') as file:
            json.dump(train_data, file)
        with open(eval_json, 'w', encoding='utf8') as file:
            json.dump(eval_data, file)

        dataset = load_dataset("json", data_files={"train": train_json, "eval": eval_json})
        return dataset





    def dicts_to_dpo_dataset(self, train_dict, eval_dict):
        df = pd.DataFrame()

        for key in train_dict:
            df[key] = train_dict[key]

        train_csv = os.path.join(TMP_DIR, 'temp_train_data.csv')
        eval_csv  = os.path.join(TMP_DIR, 'temp_eval_data.csv')
        df.to_csv(train_csv, index=False)


        df = pd.DataFrame()

        for key in eval_dict:
            df[key] = eval_dict[key]

        df.to_csv(eval_csv, index=False)

        dpo_dataset = load_dataset("csv", data_files={"train": train_csv, "eval": eval_csv})
        return dpo_dataset
