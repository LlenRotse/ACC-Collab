import numpy as np
import copy
from utility import RegExHelper




class Debate:

    def __init__(self):
        self.last_model = None

    def debate(self, debaters, question_batch, answer_batch, context_batch, question_set,
               max_rounds=10, max_trys=5, summarized=False, router=None,
               use_judge=False,
               verbose=False):

        # holds all responses for this batch, over all rounds
        # shape gets fixed function s.t. the return shape is (batch_size, num_rounds, num_models)
        prompt_batch_all = [] # shape = (num_rounds, num_models, batch_size)
        resp_batch_all   = [] # shape = (num_rounds, num_models, batch_size)
        pred_batch_all   = [] # shape = (num_rounds, num_models, batch_size)


        # holds all responses at the previous round of debate (equivalent to all_resp_batch[:, r-1, :])
        previous_resp_batch = [] # shape = (batch_size, num_models)


        ########################
        ### main debate loop ###
        for r in range(max_rounds):
            print(f'round={r}    ', end='\r')
            if verbose > 0:
                print(f'.     ROUND {r}')

            prompt_batch_round = [] # shape = (num_models, batch_size)
            resp_batch_round   = [] # shape = (num_models, batch_size)
            pred_batch_round   = [] # shape = (num_models, batch_size)

            non_support_resps = []
            for jj, debater in enumerate(debaters):
                # prompt_batch.shape == resp_batch.shape == pred_batch.shape == batch_size
                prompt_batch, resp_batch, pred_batch = debater.batch_query(r, question_batch, context_batch, question_set,
                                                                           previous_resp_batch=previous_resp_batch,
                                                                           max_trys=max_trys, verbose=verbose,
                                                                           non_support_resps=non_support_resps, 
                                                                           answer_batch=answer_batch,
                                                                           )

                print('#'*100)
                print(f"Debater {jj}, support={debater.is_support}")
                print(f'MODEL_PATH={debater.saved_model_path}')
                print(f'MODEL_PATH={getattr(debater.model, "saved_model_path", "none")}')
                print("#"*100)
                
                if not debater.is_support:
        
                    non_support_resps.append(resp_batch) # shape = (num_models, batch_size)

                prompt_batch_round.append(prompt_batch)
                resp_batch_round.append(resp_batch)
                pred_batch_round.append(pred_batch)

                if debaters[jj].saved_model_path  != debaters[(jj+1)%len(debaters)].saved_model_path:
                    debaters[jj].model.deload_model()


            # transposing resp_batch_round such that it is the correct shape for the debate prompt

            previous_resp_batch = [[resp_batch_round[m][b] for m in range(len(debaters))] for b in range(len(question_batch))]

            prompt_batch_all.append(prompt_batch_round)
            resp_batch_all.append(resp_batch_round)
            pred_batch_all.append(pred_batch_round)


        # fixing shapes to be (batch_size, num_rounds, num_models)
        prompt_batch_all = [[[prompt_batch_all[r][m][b] for m in range(len(debaters))]
                                                        for r in range(max_rounds)]
                                                        for b in range(len(question_batch))]

        resp_batch_all    = [[[resp_batch_all[r][m][b]  for m in range(len(debaters))]
                                                        for r in range(max_rounds)]
                                                        for b in range(len(question_batch))]

        pred_batch_all    = [[[pred_batch_all[r][m][b]  for m in range(len(debaters))]
                                                        for r in range(max_rounds)]
                                                        for b in range(len(question_batch))]

        # creates judgments for datasets which do not have ground truth
        judgement_batch_all = None
        if use_judge and question_set.Qtype == 'TruthQ':
            judgement_batch_all = [[[None  for m in range(len(debaters))]
                                           for r in range(max_rounds)]
                                           for b in range(len(question_batch))]
            for r in range(max_rounds):
                for m in range(len(debaters)):
                    if debaters[m].is_support:
                        judgement_batch = [None]*len(question_batch)
                    else:
                        judgement_batch = question_set.judge_batch(np.array(resp_batch_all)[:, r, m], question_batch, answer_batch)
                    for b in range(len(question_batch)):
                        judgement_batch_all[b][r][m] = judgement_batch[b]

        print()
        return prompt_batch_all, resp_batch_all, pred_batch_all, judgement_batch_all


    def display_acc(self, d, max_rounds=5, question_set=None):
        """
        Displays (prints) the average accuracy (for each round) of debate.
        Computed via average accuracy all non-support debaters
        :param d: dictionary storing the debate data
        :param max_rounds: number of debate rounds
        :param Qtype:
        :return: None
        """
        
        keys = np.array(list(d.keys()))
        if len(keys.shape)==1:
            num_questions = len(d)
            num_trials    = 1
        elif len(keys.shape)==2:
            num_questions = len(set(keys[:,0]))
            num_trials    = len(set(keys[:,1]))

        is_not_support = (1 - np.array([d[key]['is_support'] for key in d])).astype(bool)  # shape = (num_questions*num_trials, num_models)

        all_preds = np.array([d[key]['all_preds'] for key in d], dtype='str') # shape = (num_questions*num_trials, num_rounds, num_models)
        # BoolQ answers are bools (True/False) while predictions are ints (1/0);
        # normalize bools to 1/0 so the string comparison below matches.
        answers   = np.array([int(d[key]['answer']) if isinstance(d[key]['answer'], bool) else d[key]['answer']
                              for key in d], dtype='str') # shape = (num_questions*num_trials, )

        max_rounds = all_preds.shape[1]
        acc = np.zeros((len(d), max_rounds)) # shape = (num_questions*num_trials, num_rounds)

        for i in range(len(answers)):
            for r in range(max_rounds):
                q_acc = (all_preds[i, r] == answers[i])[is_not_support[i]]
                acc[i][r] = np.mean(q_acc)


        if num_trials > 1:
            trial_accs = np.zeros((max_rounds, num_trials))
            for t in range(num_trials):
                trial_key_idx = keys[:,1]==t

                trial_accs[:, t] = np.mean(acc[trial_key_idx], axis=0)

            conf_acc = 1.96*trial_accs.std(axis=1)/np.sqrt(num_trials)

        mn_acc = np.mean(acc, axis=0)
        print(f'###### Questions={num_questions}, Trials={num_trials} ####')
        # print(f'       Qtype={Qtype}')

        print('Acc:  [', end='')
        for i, a in enumerate(mn_acc):
            print(f' {round(a, 4)}', end=', ')
        print(']')
        if num_trials > 1:
            print('Conf: [', end='')
            for i, a in enumerate(conf_acc):
                print(f' {round(a, 4)}', end=', ')
            print(']')
            print()
       
















class Debater:
    """
    Object for a debater (aka an agent).
    Debaters have an LLM and a persona (aka a role)


    To add a new role to a debater we must
        1) add an if-statement to Debater.batch_query()
        2) add a prompt to the correct Question class (question_set)
    """

    def __init__(self, model_name, model_params, question_set, llm_helper, role='none', is_support=False, saved_model_path='None', temp=None):
        self.llm_helper = llm_helper
        self.reg = RegExHelper()

        self.model_name   = model_name
        self.role         = role
        self.is_support   = is_support
        self.saved_model_path = saved_model_path

        self.model_params = copy.deepcopy(model_params)
        if temp is not None:
            self.model_params['temperature'] = temp
        

        self.model, self.model_params = self.llm_helper.load_llm(self.model_name, self.model_params, load_redundant_models=False, saved_model_path=saved_model_path)

        self.question_set = question_set




    def batch_query(self, r, question_batch, context_batch, question_set, previous_resp_batch=None,
                    max_trys=5, verbose=False, non_support_resps=None, answer_batch=None):
        """
        prompts the models based on the current debate round and their assigned persona
        :param r: debate round
        :param question_batch: batch of questions
        :param context_batch: context associated with the questions (e.g., passages in BoolQ)
        :param previous_resp_batch: batch of responses from the previous timestep
        :param verbose:
        :return: a batch of responses
        """

        # at the first round of debate (r==0) all models are given a "normal" prompt
        if non_support_resps is None or len(non_support_resps) == 0 or not self.is_support:
            resps_to_use = previous_resp_batch
        else:
            resps_to_use = [[non_support_resps[m][ii] for m in range(len(non_support_resps))] for ii in range(len(non_support_resps[0]))] # size = (batch_size, models)


        if r == 0 and not self.is_support:
            prompt_batch = self.question_set.batch_prompt(self.question_set.basic_prompt, [{'question': question, 'context': context} for question, context
                                                    in zip(question_batch, context_batch)])            


        # each of the next blocks handles prompts for each type of persona
        elif self.role == 'none':
            prompt_batch = self.question_set.batch_prompt(self.question_set.debate_prompt, [
                {'responses': responses, 'question': question, 'context': context,} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'judge':
            prompt_batch = self.question_set.batch_prompt(self.question_set.judge_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'advocate':
            prompt_batch = self.question_set.batch_prompt(self.question_set.devils_advocate_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'context':
            prompt_batch = self.question_set.batch_prompt(self.question_set.context_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'detail':
            prompt_batch = self.question_set.batch_prompt(self.question_set.detail_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'ask_question':
            prompt_batch = self.question_set.batch_prompt(self.question_set.question_ask_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'answer_question':
            prompt_batch = self.question_set.batch_prompt(self.question_set.question_answer_prompt, [
                {'question': question, 'clarifying_questions': clarifying_questions, 'context': context} for
                responses, question, context, clarifying_questions in
                zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'refute':
            prompt_batch = self.question_set.batch_prompt(self.question_set.refute_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])

        elif self.role == 'helper':
            prompt_batch = self.question_set.batch_prompt(self.question_set.helper_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])
        elif self.role == 'test-persona':
            prompt_batch = self.question_set.batch_prompt(self.question_set.test_persona_prompt, [
                {'responses': responses, 'question': question, 'context': context} for responses, question, context
                in zip(resps_to_use, question_batch, context_batch)])
        elif self.role == '1oracle':
            prompt_batch = self.question_set.batch_prompt(self.question_set.targeted_detail_prompt, [
                {'responses': responses, 'question': question, 'context': context, 'answer': answer, 'correct': True} for responses, question, context, answer
                in zip(resps_to_use, question_batch, context_batch, answer_batch)])
        elif self.role == '0oracle':
            prompt_batch = self.question_set.batch_prompt(self.question_set.targeted_detail_prompt, [
                {'responses': responses, 'question': question, 'context': context, 'answer': answer, 'correct': False} for responses, question, context, answer
                in zip(resps_to_use, question_batch, context_batch, answer_batch)])
        else:
            print(f'role=\"{self.role}\" is an invalid role')
            exit()

        if self.is_support:
            _, resp_batch = self.model.batch_query(prompt_batch, **self.model_params)
            pred_batch = [None] * len(resp_batch)

        else:
             batch_to_use = context_batch
             if question_set.Qtype == 'BBH':
                 batch_to_use = answer_batch
             resp_batch, pred_batch = self.batch_answers_with_validation(prompt_batch, question_set,
                                                                         context_batch=batch_to_use,
                                                                         max_trys=max_trys, verbose=verbose)



        # for debug
        if verbose > 1:
            print('#'*100)
            print(f'    round={r}')
            print(f'    role={self.role}')
            for response, prompt in zip(resp_batch, prompt_batch):
                print(f'     prompt={prompt}')
                print()
                print(f'     response={response}')
                print()
            print('#'*100)


        # outputs are mostly for debugging
        return prompt_batch, resp_batch, pred_batch


    def batch_answers_with_validation(self, prompt_batch,  question_set, max_trys=5, context_batch=None, verbose=False):
        """
        Gets a batches of responses from an LLM debater and ensures that they are valid responses
        This function iteratively prompts the model until all responses to prompt_batch are valid
        A response is valid if it "answers" the question
        :param prompt_batch: batch of prompts to pass to model
        :param question_set: question_set of the current dataset
        :param max_trys: maximum number of times to retry invalid responses
        :param context_batch: context (e.g., passage in BoolQ, options in MMLU, etc.)
        :param verbose:
        :return: a batch of responses which have been checked for validity
        """

        _, response_batch = self.model.batch_query(prompt_batch, **self.model_params)

        # extracts the model's predictions from its response (e.g., Yes or No in BoolQ)
        preds = self.reg.get_batch_answer(response_batch, question_set.Qtype, contexts=context_batch)

        # if a response contains no valid answer, get_batch_answer returns None
        # we save all invalid responses inorder to reprompt models which have provided invalid responses
        invalid_idx = [ii for ii in range(len(preds)) if preds[ii] == None]

        trys = 0
        # this block handles the reprompting
        for trys in range(max_trys):
            if len(invalid_idx) == 0:
                break

            # for debug
            if verbose > 0:
                print(f'batch_answer_with_validation(), trys={trys}, len(invalid_idx)={len(invalid_idx)}')
                if verbose > 1:
                    print('##### invalid responses #####')
                    for idx in invalid_idx:
                        print(f'    idx={idx}) response={response_batch[idx]}')
                        print()


            new_batch = [copy.deepcopy(prompt_batch[ii]) for ii in invalid_idx]
            _, new_responses = self.model.batch_query(new_batch, **self.model_params)
            new_preds = self.reg.get_batch_answer(new_responses, question_set.Qtype, contexts=context_batch)

            valid_idx = [ii for ii, pred in zip(invalid_idx, new_preds) if pred != None]

            for ii, new_response, new_pred in zip(invalid_idx, new_responses, new_preds):
                if ii in valid_idx:
                    response_batch[ii] = new_response
                    preds[ii] = new_pred
                else:
                    pass
            invalid_idx = [ii for ii in invalid_idx if ii not in valid_idx]

            trys += 1

        # if max_trys has been reached and the model still has not given a valid answer
        # then we manually assign them a "random" answer depending on the data
        # not that for openended, or judge based tasks, this is not necessary
        ############################
        ## we might want to remove this for training (this is more of a trick to guarantee that no model is ever
        ## worse than chance in debate
        ##############################
        for j in range(len(preds)):
            if preds[j] == None:
                if question_set.Qtype == 'BoolQ':
                    preds[j] = 0.5
                elif question_set.Qtype in ['MMLUQ', 'BBH', 'SCIQ', 'MedMCQA', 'ARC']:
                    #preds[j] = random.choice(['A', 'B', 'C', 'D'])
                    preds[j] = 'Z'
                else:
                    preds[j] = -1
        pred_batch = preds

        return response_batch, pred_batch




