
from llms import GPT, Llama2, FlanT5, FlanAlpaca, LLMHelper
import utility

import pandas as pd
import numpy as np

import torch
from questions import BoolQuestion, MMLUQuestion, ArithmaticQuestion, GSMQuestion, BBHQuestion, ARCQuestion, MedMCQAQuestion, SCIQQuestion
from debaters import Debate, Debater
import time
import os
import argparse
import copy
import time
import random




def debate_on_dataset(model_names, params, roles, support_list, num_rounds, question_set, num_questions, q_start, batch_size, output_dir, resume=False, use_judge=False, saved_model_list='', bytenas=None, temps=None, max_trys=2, num_trials=1):
    """
    Run debate on a specified dataset
    All outputs and prompts are stored and saved in a dict.
    :param model_names: list of model names for each debater to use
    :param roles: list of personas for each debater
    :param support_list: list of booleans indicating whether a given debater is serving a supporting role
    :param num_rounds: number of rounds in debate
    :param question_set: dataset of questions
    :param num_questions: number of total questions to answer via debate
    :param batch_size: number of debates to run in parallel
    :param resume: boolean, if True, then we load previously answered questions from memory
    :param use_judge: boolean, if True, calls GPT-4 as a judge (only used when the dataset does not have ground truth)
    :return: Nothing
    """
    # loads all questions, answer, and context (e.g., passages in BoolQ) from the dataset
    all_questions = [(question, answer, context) for question, answer, context in question_set.gen_question()]
    
    if   question_set.hold_out_subjects is None and question_set.hold_in_subjects is None:
        idx = list(range(len(all_questions)))
    elif question_set.hold_in_subjects is not None:
        idx = [i for i in range(all_questions) if question_set.catas[i] in question_set.hold_in_subjects]
    elif question_set.hold_out_subjects is not None:
        idx = [i for i in range(all_questions) if question_set.catas[i] not in question_set.hold_out_subjects]

    all_questions = [all_questions[i] for i in idx]
    all_questions = all_questions[q_start: q_start+num_questions]

    # logic to create batches
    num_questions = min(num_questions, len(all_questions))
    num_epochs = num_questions // batch_size
    if num_questions % batch_size != 0:
        num_epochs += 1


    # loads all models and their corresponding parameters
    # will not load duplicate models (provides a "pointer" instead)
    
    
    debaters = [Debater(model_name, params, question_set, llm_helper, role=role, is_support=is_support, saved_model_path=saved_model_path, temp=temp)
                        for model_name, role, is_support, saved_model_path, temp in zip(model_names, roles, support_list, saved_model_list, temps)]

    # if not os.path.exists(f'{output_dir}/{question_set.Qtype}/results/debate'):
    #     os.makedirs(f'{output_dir}/{question_set.Qtype}/results/debate')
    # might make ugly file names, but it will make these files much easier to load in the future
    # save_name = (f'{output_dir}/{question_set.Qtype}/results/debate/model_names={model_names}_roles={roles}_support_list={support_list}_'
    #              f'rounds={num_rounds}_num_questions={num_questions}')
    

  
    role_tag = '-'.join([role[:2] for role in roles])
    model_tag = ''
    for model_name in model_names:
        model_tag += model_name
        
    save_name = f'{output_dir}/roles={role_tag}_r={num_rounds}_q={q_start}-{q_start+num_questions}'
    print(f'SAVE_NAME={save_name}')
    folder_name = output_dir
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    

    d = {} # holds debate results
    # loads previously completed debate rounds
    if resume and utility.load_dict(save_name) is not None:
        loaded_d = utility.load_dict(save_name)
        for i in loaded_d:
            d[i] = loaded_d[i]
        print()
        print(f'Loaded {len(d)} questions from {save_name}')
        print()
        time.sleep(5)

    
    #####################################
    ######## debate block ###############
    #####################################
    for trial in range(num_trials):
        i = 0
        for epoch in range(num_epochs):

            # first checks d (which may have been loaded via resume=True)
            # and skips all batches which are already in d
            skip = True
            for j in range(batch_size):
                if i + j not in d:
                    skip = False
            if skip:
                if verbose > 0:
                    print(f'    ****** Skipping {i} to {i + batch_size} *****')
                i += batch_size
                continue

            batch     = all_questions[batch_size * epoch: batch_size * (epoch + 1)]
            batch_idx = idx[batch_size * epoch: batch_size * (epoch + 1)]

            question_batch = [question for question, answer, context in batch]
            answer_batch   = [answer   for question, answer, context in batch]
            context_batch  = [context  for question, answer, context in batch]

            # each has shape (batch_size, num_rounds, num_models)
            prompt_batch_all, resp_batch_all, pred_batch_all, judgement_batch_all = debate.debate(debaters,
                                                                            question_batch, answer_batch, context_batch,
                                                                            question_set, use_judge=use_judge,
                                                                            max_rounds=num_rounds, max_trys=max_trys, summarized=False,
                                                                            router=None, verbose=False)
            # for resp_batch in resp_batch_all:
            #     for ii, resps in enumerate(resp_batch):
            #         print(f'#### ROUND {ii} ####')
            #         for jj, resp in enumerate(resps):
            #             print(f" MODEL {jj}")
            #             print('     ', resp)


            # saving outputs of debate
            for b in range(len(batch)):
                # i keeps track of the current question number (not the current batch index)
                d[(i, trial)] = {'question': question_batch[b], 'answer': answer_batch[b], 'context': context_batch[b],
                        'all_prompts': prompt_batch_all[b],  'all_resps': resp_batch_all[b], 'all_preds': pred_batch_all[b]}

                d[(i, trial)]['model_names'] = model_names
                d[(i, trial)]['is_support'] = [debater.is_support for debater in debaters]
                d[(i, trial)]['roles']      = [debater.role       for debater in debaters]

                d[(i, trial)]['question_id'] = batch_idx[b]

                if question_set.Qtype == 'TruthQ' and use_judge:
                    d[(i, trial)]['all_evaluations'] = judgement_batch_all[b]
                    d[(i, trial)]['all_preds'] = judgement_batch_all[b]
                    d[(i, trial)]['answer'] = 1

                i += 1

            debate.display_acc(d, num_rounds, question_set)
            ################
            # Display invalid responses
            if question_set.Qtype=='TruthQ':
                num_invalid = 0
                num_total   = 1
            else:
                num_invalid = sum(debater.reg.invalid_query for debater in debaters)
                num_total   = sum(debater.reg.total_query for debater in debaters)
            print(f"    TOTAL: {num_total}, INVALID: {num_invalid}, RATIO: {num_invalid/num_total}")
            # save_name = save_name.replace(' ', '_').replace('\'', '').replace(',', '_')
            utility.save_dict(d, save_name)



def gen_answerer_traget_data(model_names, params, roles, support_list, num_rounds, question_set, num_questions, q_start, batch_size, output_dir, resume=False, use_judge=False, saved_model_list='', bytenas=None, temps=None, max_trys=2, correct=True, project_name=None, debate_q_end=-1):
    model_tag = '__'.join(sorted(set(model_names)))
    role = roles[1][:2]

    # Number of questions in the base debate file to load. Defaults to the full
    # dataset size used in the paper, but can be overridden via --debate_q_end
    # so that small local runs can find the file they just generated.
    dataset_num_questions = {'BoolQ': 9000, 'MMLUQ': 15_500, 'SCIQ': 11_000, 'BBH': 5_000, 'MedMCQA': 10_000, 'ARC': 4000}
    q_end = str(debate_q_end) if debate_q_end and debate_q_end > 0 else str(dataset_num_questions[question_set.Qtype])

    d = utility.load_dict(f'{bytenas}/{project_name}/{question_set.Qtype}/{model_tag}/roles=no-{role}_r={num_rounds}_q=0-{q_end}', verbose=False, key_type='int_tuple')
    max_loading_attempts = 10
    loading_attempts = 0
    while d is None:
        if loading_attempts >= max_loading_attempts:
            exit(-100)
        loading_attempts += 1
        t = random.randint(10, 300)
        time.sleep(t)
        d = utility.load_dict(f'{bytenas}/{project_name}/{question_set.Qtype}/{model_tag}/roles=no-{role}_r={num_rounds}_q=0-{q_end}', verbose=False, key_type='int_tuple')

    keys = np.array(list(d.keys()))
    max_t = max(keys[:,1])
    num_trials = max_t+1
    # num_trials=1
    
    # new_d = {(i-q_start, t): d[(i, t)] for i in range(q_start, q_start+num_questions) for t in range(num_trials)}
    new_d = {}

    debater = Debater(model_names[0], params, question_set, llm_helper, role=roles[0], is_support=support_list[0], saved_model_path=saved_model_list[0], temp=temps[0])

    for t in range(num_trials):
        for r in range(-1, num_rounds-1):
            target_prompts = []
            idx_set = [] # keeps track of all keys that where used this round
            for i in range(q_start, q_start+num_questions):
                # incase we decide not to save all trials
                if (i, t) not in d:
                    continue
                idx_set.append((i, t))
                question       = d[(i, t)]['question']
                context        = d[(i, t)]['context']
                answer         = d[(i, t)]['answer']
                
                if r==-1:
                    target_prompts.append(question_set.targeted_basic_prompt(question=question, context=context, answer=answer, correct=correct))
                else:
                    previous_resps = d[(i, t)]['all_resps'][r]
                    target_prompts.append(question_set.targeted_debate_prompt(question=question, context=context, answer=answer, responses=previous_resps, correct=correct))
            
            resps, preds = debater.batch_answers_with_validation(target_prompts,  question_set, max_trys=max_trys, context_batch=None, verbose=False)

            for i, t in idx_set:
                new_key = (i-q_start, t) # offset by q_start so that all saved dictonaries are always zero-indexed
                if new_key not in new_d:
                    new_d[new_key] = copy.deepcopy(d[(i, t)])
                new_d[new_key]['all_resps'][r+1][0] = resps[new_key[0]]
                new_d[new_key]['all_preds'][r+1][0] = preds[new_key[0]]
            
            save_name = f'{bytenas}/{project_name}/{question_set.Qtype}/{model_tag}/target_c={correct}_{role}_r={num_rounds}_q={q_start}-{q_start+num_questions}'
            print(f'Trial={t}, Round={r}')
            print(f'Saving dict to {save_name}')

            utility.save_dict(new_d, save_name)
        

def gen_support_traget_data(model_names, params, roles, support_list, num_rounds, question_set, num_questions, q_start, batch_size, output_dir, resume=False, use_judge=False, saved_model_list='', bytenas=None, temps=None,
                            max_trys=2, correct=1, project_name=None, debate_q_end=-1):

    model_tag = '__'.join(sorted(set(model_names)))
    role = roles[1][:2]
    dataset_num_questions = {'BoolQ': 9000, 'MMLUQ': 15_500, 'SCIQ': 11_000, 'BBH': 5_000, 'MedMCQA': 10_000, 'ARC': 4000}
    q_end = str(debate_q_end) if debate_q_end and debate_q_end > 0 else str(dataset_num_questions[question_set.Qtype])

    d = utility.load_dict(f'{bytenas}/{project_name}/{question_set.Qtype}/{model_tag}/roles=no-{role}_r={num_rounds}_q=0-{q_end}', verbose=False, key_type='int_tuple')
    max_loading_attempts = 10
    loading_attempts = 0
    while d is None:
        if loading_attempts >= max_loading_attempts:
            exit(-100)
        loading_attempts += 1
        t = random.randint(10, 300)
        time.sleep(t)
        d = utility.load_dict(f'{bytenas}/{project_name}/{question_set.Qtype}/{model_tag}/roles=no-{role}_r={num_rounds}_q=0-{q_end}', verbose=True, key_type='int_tuple')
        
    keys = np.array(list(d.keys()))

    max_t = max(keys[:,1])
    num_trials = max_t+1
    
    new_d = {}
    
    print('######\n'*5, 'max_t', max_t, '\n######'*5)
    
    debater = Debater(model_names[0], params, question_set, llm_helper, role=roles[0], is_support=support_list[0], saved_model_path=saved_model_list[0], temp=temps[0])
    support = Debater(model_names[1], params, question_set, llm_helper, role=roles[1], is_support=support_list[1], saved_model_path=saved_model_list[1], temp=temps[1])
    
    # generate targeted assistant responses
    for t in range(num_trials):
        
        for r in range(num_rounds):
            answers = []
            questions = []
            contexts = []
            target_prompts = []
            idx_set = [] # keeps track of all keys that where used this round
            for i in range(q_start, q_start+num_questions):
                if (i, t) not in d:
                    continue
                idx_set.append((i, t))
                question       = d[(i, t)]['question']
                context        = d[(i, t)]['context']
                answer         = d[(i, t)]['answer']
                previous_resps = d[(i, t)]['all_resps'][r]
                
                answers.append(answer)
                questions.append(question)
                contexts.append(context)
                
                # currently only implemented for detail persona
                if t > 0:
                    target_prompts.append(question_set.targeted_detail_prompt(question=question, context=context, answer=answer, responses=previous_resps, correct=correct, strategy=t))
            if t > 0:
                output, sup_resps = support.model.batch_query(target_prompts, **support.model_params)
            for i, t in idx_set:
                new_key = (i-q_start, t) # offset by q_start so that all saved dictonaries are always zero-indexed
                if new_key not in new_d:
                    new_d[new_key] = copy.deepcopy(d[(i, t)])
                if t > 0:
                    new_d[new_key]['all_resps'][r][1] = sup_resps[new_key[0]]

    # deload the support only if we need to
    if support.model_name != debater.model_name:
        support.model.deload()
    # evaluate the quality of the assistant responses
    for t in range(num_trials):
        
        for r in range(num_rounds):
            answers = []
            questions = []
            contexts = []
            prompts = []
            idx_set = [] # keeps track of all keys that where used this round
            for i in range(q_start, q_start+num_questions):
                if (i, t) not in d:
                    continue
                new_key = (i-q_start, t)
                idx_set.append((i, t))
                question       = d[(i, t)]['question']
                context        = d[(i, t)]['context']
                answer         = d[(i, t)]['answer']
                previous_resps = new_d[new_key]['all_resps'][r]
                
                answers.append(answer)
                questions.append(question)
                contexts.append(context)


                prompts.append(question_set.debate_prompt(question=question, context=context, responses=previous_resps))
            
            all_preds = []
            for jj in range(10):
                resps, preds = debater.batch_answers_with_validation(prompts,  question_set, max_trys=max_trys, context_batch=None, verbose=False)
                all_preds.append(preds)
            all_preds = np.array(all_preds) # shape = (10, num_questions)
            answers = np.array(answers)
            avg_accs  = np.mean(all_preds==answers, axis=0) # shape = (num_questions, )

            for i, t in idx_set:
                new_key = (i-q_start, t) # offset by q_start so that all saved dictonaries are always zero-indexed
                if 'avg_accs' not in new_d[new_key]:
                    new_d[new_key]['avg_accs'] = [[0, 0] for _ in range(num_rounds)]
                new_d[new_key]['avg_accs'][r][0] = avg_accs[new_key[0]]
            
            save_name = f'{bytenas}/{project_name}/{question_set.Qtype}/{model_tag}/sup_t_c={correct}_{role}_r={num_rounds}_q={q_start}-{q_start+num_questions}'
            print(f'Trial={t}, Round={r}')
            print(f'Saving dict to {save_name}')

            utility.save_dict(new_d, save_name)
        











if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--Qtype",         type=str, help="Name of the dataset from which questions are constructed. Valid values: BoolQ, MMLUQ, MathQ, TruthQ")
    parser.add_argument("--num_questions", type=int, help="Number of question to be asked (clipped to size of the dataset)")
    parser.add_argument("--num_rounds",    type=int, help="Number of rounds of debate.")
    parser.add_argument("--batch_size",    type=int, help="Number of debates to conduct in parallel")

    # program with do questions q_start through q_start+num_questions
    parser.add_argument("--q_start", type=int, help="index of the question to start at", default=-1)

    parser.add_argument("--debate_type", type=str,  help="Must be either \"debate\" or \"pref\"")

    parser.add_argument("--roles",        type=str,  help="A list of roles (personas) for the debaters. The length of this list implies the number of debaters.")
    parser.add_argument('--temps',        type=str,  help='list of temps to use for each LLM')
    parser.add_argument('--max_trys',     type=int,  help='number of times to retry answering LLM')
    parser.add_argument("--model_names",  type=str,  help="A list of model names which are used by the debaters.")
    parser.add_argument("--support_list", type=str,  help="A list of flags indicating whether a given debater is serving a support role.")
    parser.add_argument("--saved_model_list", type=str,  help="A list of model dirs on where saved LORA models are stored, if None default model will be given.")
    
    parser.add_argument('--correct',    type=int,  help='1 to use target_answer=correct_answer,  0 for target_answer=incorrect_answer')
    parser.add_argument('--num_trials', type=int,  help='number of trials')

    parser.add_argument("--use_judge", type=lambda x: x=='True', default=False, help="Boolean indicator, if true, then an LLM judge is used to evaluate correctness. Currently only supported for TruthQ")
    parser.add_argument("--resume",    type=lambda x: x=='True', default=False, help="Boolean indicator, if true, then previously completed questions are loaded (and skipped). Useful instances where your job may be killed mid-run.")

    parser.add_argument("--output_dir",   type=str, help="Output director for example: ./")
    parser.add_argument("--bytenas",   type=str, default="./storage",
                        help="Local storage root where generated debate/target data is written and read "
                             "(paths are <bytenas>/<project_name>/<Qtype>/<model_tag>/...). Any local directory works.")
    parser.add_argument("--arnold",   type=utility.str2bool, nargs="?", const=True, default=False, help="Running on arnold?")

    parser.add_argument("--hold_out_subjects",type=str, help="for MMLUQ a list of subjects to NOT use in MMLU eval", default=None)
    parser.add_argument("--hold_in_subjects", type=str, help="for MMLUQ a list of subjects to use in MMLU eval", default=None)

    parser.add_argument("--project_name", type=str)

    parser.add_argument("--debate_q_end", type=int, default=-1,
                        help="For debate_type=target/sup_target: number of questions in the base debate file to load "
                             "(roles=no-<role>_r=<rounds>_q=0-<debate_q_end>). Defaults (-1) to the full paper dataset size; "
                             "set it to the num_questions used in your debate run for small local runs.")

    args = parser.parse_args()


    # Objects used to run the debate
    llm_helper = LLMHelper(arnold=args.arnold, bytenas=args.bytenas)
    reg = utility.RegExHelper()
    debate = Debate()

    # creates a Question object for a given dataset
    # each dataset requires its own Question object to be written
    question_type_dict = {'BoolQ':   BoolQuestion,
                          'MMLUQ':   MMLUQuestion,
                          'MathQ':   ArithmaticQuestion,
                        #   'TruthQ': TruthfulQuestion,
                          'GSMQ':    GSMQuestion,
                          'BBH':     BBHQuestion,
                          'SCIQ':    SCIQQuestion,
                          'MedMCQA': MedMCQAQuestion,
                          'ARC':     ARCQuestion,
                          }

    if args.Qtype == 'MMLUQ':
        if args.hold_out_subjects == 'None' or args.hold_out_subjects == '':
            args.hold_out_subjects = None
        else:
            args.hold_out_subjects = args.hold_out_subjects.split(',')

        if args.hold_in_subjects == 'None' or args.hold_in_subjects == '':
            args.hold_in_subjects = None
        else:
            args.hold_in_subjects = args.hold_in_subjects.split(',')

        question_set  = question_type_dict[args.Qtype](hold_out_subjects=args.hold_out_subjects, hold_in_subjects=args.hold_in_subjects)
    else:
        question_set  = question_type_dict[args.Qtype]()

    for arg in vars(args):
        print(arg, getattr(args, arg))
    assert args.use_judge == (args.Qtype in ['TruthQ'])

    # default params for the models (we probably don't want to change this).
    params = {'temperature': 0.8,
              'max_tokens': 1024,
              'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              }

    # verbose = 0: only very basic print-outs.
    # verbose = 1: prints out debate information without prompts and responses (e.g., number of invalid responses).
    # verbose = 2: prints out all information including prompts and responses.
    verbose = 0
        
    args.model_names = args.model_names.split(',')
    args.roles = args.roles.split(',')
    args.saved_model_list = args.saved_model_list.split(',')

    args.support_list = [a=='True' for a in args.support_list.split(',')]

    temps = [float(temp) for temp in args.temps.split(',')]

    args.roles = args.roles[:len(args.model_names)]
    args.support_list = args.support_list[:len(args.model_names)]

    if args.debate_type=='debate':
        debate_on_dataset(args.model_names, params, args.roles, args.support_list, args.num_rounds, question_set,
                        args.num_questions, args.q_start, args.batch_size, resume=args.resume, use_judge=args.use_judge, output_dir=args.output_dir, saved_model_list=args.saved_model_list, bytenas=args.bytenas, temps=temps, max_trys=args.max_trys,
                        num_trials=args.num_trials,
                        )

    elif args.debate_type=='target':
        gen_answerer_traget_data(args.model_names, params, args.roles, args.support_list, args.num_rounds, question_set,
                        args.num_questions, args.q_start, args.batch_size, resume=args.resume, use_judge=args.use_judge, output_dir=args.output_dir, saved_model_list=args.saved_model_list, bytenas=args.bytenas, temps=temps, max_trys=args.max_trys,
                        correct=args.correct, project_name=args.project_name, debate_q_end=args.debate_q_end,
                        )
    
    elif args.debate_type=='sup_target':
        gen_support_traget_data(args.model_names, params, args.roles, args.support_list, args.num_rounds, question_set,
                        args.num_questions, args.q_start, args.batch_size, resume=args.resume, use_judge=args.use_judge, output_dir=args.output_dir, saved_model_list=args.saved_model_list, bytenas=args.bytenas, temps=temps, max_trys=args.max_trys,
                        correct=args.correct,  project_name=args.project_name, debate_q_end=args.debate_q_end,
                        )
    



# pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
# transformers==4.43







