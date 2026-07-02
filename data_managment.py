import os
import utility
import numpy as np
from questions import MMLUQuestion
import matplotlib.pyplot as plt
import copy


def compute_acc(d, max_rounds=5):
    keys = np.array(list(d.keys()))
    if 'avg_accs' in d[tuple(keys[0])].keys():
        acc = np.array([d[key]['avg_accs'] for key in d])[:,:,0]
        
    else:
        preds   = np.array([d[key]['all_preds'] for key in d])[:,:,0]
        answers = np.array([d[key]['answer']    for key in d])
        acc = np.zeros(preds.shape)
        for r in range(preds.shape[1]):
            acc[:,r] = preds[:,r]==answers

    num_trials = len(set(keys[:,1]))    
    if num_trials > 1:
        trial_accs = np.zeros((max_rounds, num_trials))
        for t in range(num_trials):
            trial_key_idx = keys[:,1]==t
            trial_accs[:, t] = np.mean(acc[trial_key_idx], axis=0)
        conf_acc = 1.96*trial_accs.std(axis=1)/np.sqrt(num_trials)
    
    else:
        conf_acc = np.zeros(max_rounds)
    mn_acc = np.mean(acc, axis=0)        
    
            
    return mn_acc, conf_acc


def num_invalid(d):
    preds = np.array([d[key]['all_preds'] for key in d])[:,:,0].flatten()
    return np.mean(preds==0.5) + np.mean(preds=='Z') + np.mean(preds==-1) + np.mean(preds=='z')


def trim_to_5(json_path, pattern, num_questions_per_file, q_start, q_end, key_type='int_tuple', verbose=False):
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
            

            loaded_d = utility.load_dict(json_path+file, key_type=key_type, verbose=verbose)
            if loaded_d is None:
                print(f'NOT FOUND: {json_path+file}')
                continue
            keys = np.array(list(loaded_d.keys()))
            if len(keys.shape) != 2:
                continue
            trials = len(set(keys[:,1]))
            
            new_d = {}
            for (q, t) in loaded_d:
                if t < 5:
                    new_d[(q, t)] = loaded_d[(q, t)]
                    
            utility.save_dict(new_d, json_path+file)
            

def combine_dicts(json_path, pattern, num_questions_per_file, q_start, q_end, key_type='int_tuple', verbose=False):
    
    combined_d = {}
    
    all_trials = {}
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
            
            
            loaded_d = utility.load_dict(json_path+file, key_type=key_type, verbose=verbose)
            if loaded_d is None:
                print(f'NOT FOUND: {json_path+file}')
                continue
            keys = np.array(list(loaded_d.keys()))
            if len(keys.shape) != 2:
                continue
            trials = len(set(keys[:,1]))
            
            for key in loaded_d:
                
                new_key = (key[0]+file_q_start, key[1])
                q_id = new_key[0]
                t_id = new_key[1]
                
                # check if the current question is outside (q_start, q_end)
                if q_id < q_start or q_end < q_id:
                    continue
                else:
                    combined_d[new_key] = loaded_d[key]
                
                if t_id not in all_trials:
                    all_trials[t_id] = [q_id]
                else:
                    all_trials[t_id].append(q_id)
                
    return combined_d, all_trials


def is_valid_dict(d, total_q=9000, total_t=20, d_name=None):
    keys = np.array(list(d.keys()))
    # print(f'keys.shape={keys.shape}')
    if len(keys.shape) != 2:
        return False
    if len(set(keys[:,0])) != total_q or len(set(keys[:,1])) != total_t or len(keys) != total_q * total_t:
        return False
    else:
        return True
    

def save_supports(ds0, ds1, d_reg, path, total_q, total_t):
    new_ds0 = {}
    new_ds1 = {}

    keys  = np.array(list(ds0.keys()))
    question_ids = sorted(set(keys[:,0])),
    max_t = len(set(keys[:,1]))
    max_q = len(set(keys[:,0]))
    
    if len(question_ids) != max(question_ids)+1 or len(question_ids) != total_q:
        print(f"ANSWER: len(question_ids)={len(question_ids)}, but you passed total_q={total_q}")
        return

    for q in range(max_q):
        best_t = None
        best_diff = -1
        for t in range(max_t):
            if 'avg_accs' not in ds0[(q, t)]:
                continue
            avg_acc0 = np.mean(ds0[(q, t)]['avg_accs'])
            avg_acc1 = np.mean(ds1[(q, t)]['avg_accs'])
            diff = avg_acc1 - avg_acc0
            if diff > best_diff:
                best_t, best_diff = t, diff

        new_ds0[(q, 0)] = ds0[(q, best_t)]
        new_ds1[(q, 0)] = ds1[(q, best_t)]

    print('      saving', f'{path}NewStrat_sup_t_c=0_de_r=5_q=0-{total_q}')
    print('      saving', f'{path}NewStrat_sup_t_c=1_de_r=5_q=0-{total_q}')
    
    utility.save_dict(new_ds0, f'{path}NewStrat_sup_t_c=0_de_r=5_q=0-{total_q}')
    utility.save_dict(new_ds1, f'{path}NewStrat_sup_t_c=1_de_r=5_q=0-{total_q}')
    
    
def save_answers(da0, da1, d_reg, path, total_q, dataset_name):
    
    if dataset_name == 'MMLUQ':
        question_set = MMLUQuestion()
    keys         = np.array(list(da0.keys()))
    question_ids = sorted(set(keys[:,0]))
    
    new_da0 = copy.deepcopy(da0)
    new_da1 = copy.deepcopy(da1)
    best_trials = {q: (0, 0) for q in question_ids}
    for (q, t) in d_reg:
        answer = d_reg[(q, t)]['answer']
        max_r = len(d_reg[(q, t)]['all_preds'])
        if dataset_name == 'MMLU':
            new_da0[(q, t)]['subject_type'] = question_set.catas[q]
            new_da1[(q, t)]['subject_type'] = question_set.catas[q]
        total_corrections = 0
        for r in range(max_r):
            pred  = d_reg[(q, t)]['all_preds'][r][0]
            pred0 = da0[(q, t)]['all_preds'][r][0]
            pred1 = da1[(q, t)]['all_preds'][r][0]
            if pred != answer and pred1 == answer:
                new_da0[(q, t)]['all_preds'][r][0] = pred
                new_da0[(q, t)]['all_resps'][r][0] = d_reg[(q, r)]['all_resps'][r][0]
                total_corrections += 1
                if total_corrections > best_trials[q][1]:
                    
                    best_trials[q] = (t, total_corrections)
        
    new_da0 = {(q, 0): new_da0[(q, best_trials[q][0])] for q in question_ids}
    new_da1 = {(q, 0): new_da1[(q, best_trials[q][0])] for q in question_ids}
    print('      saving', path+f'One_Trial_target_c=0_de_r=5_q=0-{total_q}')
    print('      saving', path+f'One_Trial_target_c=1_de_r=5_q=0-{total_q}')
    
    utility.save_dict(new_da0, path+f'One_Trial_target_c=0_de_r=5_q=0-{total_q}')
    utility.save_dict(new_da1, path+f'One_Trial_target_c=1_de_r=5_q=0-{total_q}')
    
    
    
def save_reg(d_reg, path, total_q):
    keys         = np.array(list(d_reg.keys()))
    question_ids = sorted(set(keys[:,0]))
    if len(question_ids) != max(question_ids)+1 or len(question_ids) != total_q:
        print(f"REG: len(question_ids)={len(question_ids)}, but you passed total_q={total_q}")
        return None
    print('      saving', path+f'roles=no-de_r=5_q=0-{total_q}')
    utility.save_dict(d_reg, path+f'roles=no-de_r=5_q=0-{total_q}')

def find_contiguous_intervals(nums):
    if not nums:
        return []

    intervals = []
    start = nums[0]
    
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1] + 1:
            # End of a contiguous interval
            if start == nums[i - 1]:
                intervals.append((start,))
            else:
                intervals.append((start, nums[i - 1]))
            start = nums[i]
    
    # Add the final interval
    if start == nums[-1]:
        intervals.append((start,))
    else:
        intervals.append((start, nums[-1]))

    return intervals
    

    
    
    
    

if __name__=='__main__':
    model_names = ["V_Llama-2-7b-chat-hf",  "Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
dataset_names = ['BoolQ', 'MMLUQ', 'SCIQ', 'MedMCQA', 'BBH', 'ARC']

dataset_num_questions = {'BoolQ': 9000, 'MMLUQ': 15_500, 'SCIQ': 11_000, 'BBH': 5_000, 'MedMCQA': 10_000, 'ARC': 4_000}

progress = {}
all_accs = {}
all_invalid = {}

all_preds = {}
all_answers = {}

num_trials = 5
for jj, dataset_name in enumerate(dataset_names):
    for ii, model_name in enumerate(model_names):
        
        num_questions = dataset_num_questions[dataset_name]
        
        path = f"{os.environ.get('LLM_COLLAB_DATA', 'training')}/{dataset_name}/{model_name}/"
        if not os.path.isdir(path):
            print(f'NOT A PATH: {path}')
            continue
        ds0, ds1 = (None, None), (None, None)
#         pattern = 'sup_t_c=0'
#         ds0 = combine_dicts(path, pattern, 500, 0, 16000, key_type='int_tuple', verbose=False)
        
#         pattern = 'sup_t_c=1'
#         ds1 = combine_dicts(path, pattern, 500, 0, 16000, key_type='int_tuple', verbose=False)
                      
        pattern = 'target_c=0'
        da0 = combine_dicts(path, pattern, 500, 0, 16000, key_type='int_tuple', verbose=False)
              
        pattern = 'target_c=1'
        da1 = combine_dicts(path, pattern, 500, 0, 16000, key_type='int_tuple', verbose=False)

        pattern = 'roles=no-de_r=5_q='
        d_reg = combine_dicts(path, pattern, 500, 0, 16000, key_type='int_tuple', verbose=False)
              
        progress[(model_name, dataset_name)] = {}
        is_valid = {}
        print(f'##### {model_name} --- {dataset_name} ######## ')
        for (d, all_trials), name in zip([ds0, ds1, da0, da1, d_reg], ['ds0', 'ds1', 'da0', 'da1', 'd_reg']):
            if d is None:
                continue
            print(f'---- {name}:  ', end = '')

            valid = is_valid_dict(d, total_q=num_questions, total_t=num_trials)
            if valid:
                print(" valid")
                is_valid[name] = True
                progress[(model_name, dataset_name)][name] = 1
                
                all_accs[(model_name, dataset_name, name)]    = compute_acc(d)
                all_invalid[(model_name, dataset_name, name)] = num_invalid(d)
            else:
                is_valid[name] = False
                total = 0
                print(" invalid")
                for key in all_trials:
                    missing = [ii for ii in range(num_questions) if ii not in all_trials[key]]
                    missing_intervals = find_contiguous_intervals(missing)
                    print(f'      {key}: {missing_intervals}')
                    total += len(all_trials[key])
                progress[(model_name, dataset_name)][name] = total/(num_questions*num_trials)
        if is_valid['d_reg']:
            save_reg(d_reg[0], path, total_q=num_questions)
            
            all_preds[(model_name, dataset_name, 'd_reg')]   = {key: d_reg[0][key]['all_preds'] for key in d_reg[0]}
            all_answers[(model_name, dataset_name, 'd_reg')] = {key: d_reg[0][key]['answer']    for key in d_reg[0]}
            
            if is_valid['da0'] and is_valid['da1']:
                save_answers(da0[0], da1[0], d_reg[0], path, total_q=num_questions)
                
                all_preds[(model_name, dataset_name, 'da0')]   = {key: da0[0][key]['all_preds'] for key in da0[0]}
                all_answers[(model_name, dataset_name, 'da0')] = {key: da0[0][key]['answer']    for key in da0[0]}

                all_preds[(model_name, dataset_name, 'da1')]   = {key: da1[0][key]['all_preds'] for key in da1[0]}
                all_answers[(model_name, dataset_name, 'da1')] = {key: da1[0][key]['answer']    for key in da1[0]}


    model_names = ["V_Llama-2-7b-chat-hf",  "Meta-Llama-3-8B-Instruct", "Mistral-7B-Instruct-v0.2"]
dataset_names = ['BoolQ', 'MMLUQ', 'SCIQ', 'MedMCQA', 'BBH', 'ARC']

dataset_num_questions = {'BoolQ': 9000, 'MMLUQ': 15_500, 'SCIQ': 11_000, 'BBH': 5_000, 'MedMCQA': 10_000, 'ARC': 4_000}

progress = {}
all_accs = {}
all_invalid = {}

num_trials = 5
for jj, dataset_name in enumerate(dataset_names):
    for ii, model_name in enumerate(model_names):
        
        num_questions = dataset_num_questions[dataset_name]
        
        path = f"{os.environ.get('LLM_COLLAB_DATA', 'training')}/{dataset_name}/{model_name}/"
        if not os.path.isdir(path):
            print(f'NOT A PATH: {path}')
            continue
        ds0, ds1 = None, None
        
#         pattern = 'sup_t_c=0'
#         ds0 = combine_dicts(path, pattern, 500, 0, 16000, key_type='int_tuple', verbose=False)
        
#         pattern = 'sup_t_c=1'
#         ds1 = combine_dicts(path, pattern, 500, 0, 16000, key_type='int_tuple', verbose=False)
                      
        file_name = f'roles=no-de_r=5_q=0-{num_questions}'
        d_reg = utility.load_dict(path+file_name, key_type='int_tuple', verbose=False)
        
        file_name = f'One_Trial_target_c=0_de_r=5_q=0-{num_questions}'
        da0 = utility.load_dict(path+file_name, key_type='int_tuple', verbose=False)
              
        file_name = f'One_Trial_target_c=1_de_r=5_q=0-{num_questions}'
        da1 = utility.load_dict(path+file_name, key_type='int_tuple', verbose=False)              


        print(f'------- {model_name} --- {dataset_name} ------- ')
        for d, name in zip([ds0, ds1, da0, da1, d_reg], ['ds0', 'ds1', 'da0', 'da1', 'd_reg']):
            if d is None:
                continue
            all_accs[(model_name, dataset_name, name)]    = compute_acc(d)
            all_invalid[(model_name, dataset_name, name)] = num_invalid(d)
        
    
        
    def plot_performance(dataset_names, model_names, all_accs, all_invalid):
        
        styles = {'d_reg': 'solid',   'da0': 'dashed',           'da1': 'solid',            'ds0': 'dashed',           'ds1': 'solid'}
        names  = {'d_reg': 'Vanilla', 'da0': 'Ans-Target (c=0)', 'da1': 'Ans-Target (c=1)', 'ds0': 'Sup-Target (c=0)', 'ds1': 'Sup-Target (c=1)'}
        colors = {'d_reg': 'black',   'da0': 'blue',             'da1': 'blue',             'ds0': 'orange',           'ds1': 'orange'}

        fig, ax = plt.subplots(len(dataset_names), len(model_names), figsize=(15, 13))
        
        xrng = range(len(list(all_accs.values())[0][0]))
        for i, dataset_name in enumerate(dataset_names):
            for j, model_name in enumerate(model_names):
                for d_name in ['d_reg', 'da0', 'da1', 'ds0', 'ds1']:
                    key = (model_name, dataset_name, d_name)
                    if key in all_accs:
                        ax[i][j].plot(xrng, all_accs[key][0], color=colors[d_name], label=names[d_name], linestyle=styles[d_name])
                        ax[i][j].fill_between(xrng, all_accs[key][0] - all_accs[key][1], all_accs[key][0] + all_accs[key][1], color=colors[d_name], alpha=0.2)
                        ax[i][j].set_xticks(range(len(all_accs[key][0])), range(len(all_accs[key][0])))
                        if i == 0:
                            ax[i][j].set_title(model_name)
                        if j == 0:
                            ax[i][j].set_ylabel(f'{dataset_name}\nAccuracy')
                        if i == len(dataset_names)-1:
                            ax[i][j].set_xlabel('Round')
                        
                        if i == j == 0:
                            ax[i][j].legend()   
        plt.savefig('fig')
        plt.show()
        
        dataset_num_questions = {'BoolQ': 9_000, 'MMLUQ': 15_500, 'SCIQ': 11_000, 'BBH': 5_000, 'MedMCQA': 10_000, 'ARC': 4_000}
        for dataset_name in dataset_names:
            for model_name in model_names:
                if (model_name, dataset_name, 'd_reg') not in all_answers or (model_name, dataset_name, 'da0') not in all_answers or (model_name, dataset_name, 'da1') not in all_answers:
                    continue
                answers = all_answers[(model_name, dataset_name, 'd_reg')]
                pr = all_preds[(model_name, dataset_name, 'd_reg')]
                p0 = all_preds[(model_name, dataset_name, 'da0')]
                p1 = all_preds[(model_name, dataset_name, 'da1')]
                
                print(dataset_name, model_name)
                for epoch in range(dataset_num_questions[dataset_name]//500):
                    ans = np.array([answers[(q, t)] for q in range(epoch*500, (epoch+1)*500) for t in range(5)])
                    
                    preds_r = np.array([pr[(q, t)] for q in range(epoch*500, (epoch+1)*500) for t in range(5)])[:,:,0]
                    preds_0 = np.array([p0[(q, t)] for q in range(epoch*500, (epoch+1)*500) for t in range(5)])[:,:,0]
                    preds_1 = np.array([p1[(q, t)] for q in range(epoch*500, (epoch+1)*500) for t in range(5)])[:,:,0]
                    
                    acc_r = np.zeros(preds_r.shape)
                    acc_0 = np.zeros(preds_0.shape)
                    acc_1 = np.zeros(preds_1.shape)
                    
                    for r in range(5):
                        acc_r[:,r] = preds_r[:,r] == ans
                        acc_0[:,r] = preds_0[:,r] == ans
                        acc_1[:,r] = preds_1[:,r] == ans
                        
                    acc_r = np.mean(acc_r, axis=0)
                    acc_0 = np.mean(acc_0, axis=0)
                    acc_1 = np.mean(acc_1, axis=0)
                    if any(acc_r[r] == acc_0[r] for r in range(5)) or any(acc_r[r] == acc_1[r] for r in range(5)):
                        print(f'         {epoch*500}-{(epoch+1)*500}')
        
        fig = plt.figure(figsize=(20, 10))
        model_invalids = {model_name: [all_invalid[(model_name, dataset_name, 'd_reg')] for dataset_name in dataset_names] for model_name in model_names}
        colors = ['red', 'blue', 'orange', 'purple']
        xrng = np.arange(len(dataset_names))
        for ii, model_name in enumerate(model_invalids):
            print(model_name)
            print(model_invalids)
            plt.bar(xrng + 0.3*ii, model_invalids[model_name], color=colors[ii], label=model_name, width=0.3)
        
        plt.ylabel('Invalid Response Fraction')
        plt.xticks(xrng, dataset_names, rotation=20)
        plt.legend()
        plt.show()

    plot_performance(dataset_names, model_names, all_accs, all_invalid)            
                