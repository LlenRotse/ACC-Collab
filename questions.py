import random
import pandas as pd
import numpy as np
import re
import json
import os
# from evaluate import load
# import evaluate

import itertools as itr
from utility import RegExHelper
# from llms import GPT

SEED = 101


class Question:
    """
    Generic question object
    Question objects are responsible for
        1) holding all data for a given dataset
        2) formatting that data into questions which can be answered by an LLM
        3) generating all prompts for a given dataset (e.g., debate prompts, persona prompts, etc.)
    """
    def __init__(self):
        self.questions = None
        self.answers = None

        self.question_type = None
        self.answer_format = None

        self.reg = RegExHelper()

    def summarize_prompt(self, responses, question):
        pass

    def batch_prompt(self, prompt_function, prompt_params):
        prompts = []
        for params in prompt_params:
            prompts.append(prompt_function(**params))
        return prompts

    def gen_question(self, num_questions):
        for i in range(num_questions):
            yield None


class BoolQuestion(Question):

    def __init__(self, hold_in_subjects=None, hold_out_subjects=None):
        super().__init__()

        # loading and formatting data
        
        self.questions = []
        self.answers   = []
        self.passages  = []
        with open('BoolQ/train.jsonl', 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file if line.strip()]
            self.questions += [d['question'].capitalize() for d in data]
            self.answers   += [d['answer'] for d in data]
            self.passages  += [self.reg.process(d['passage']) for d in data]

        with open('BoolQ/dev.jsonl', 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file if line.strip()]
            self.questions += [d['question'].capitalize() for d in data]
            self.answers   += [d['answer'] for d in data]
            self.passages  += [self.reg.process(d['passage']) for d in data]

        self.hold_in_subjects  = hold_in_subjects
        self.hold_out_subjects = hold_out_subjects
        # balancing the questions
        # pos_idx = [i for i, ans in enumerate(self.answers) if ans == True]
        # neg_idx = [i for i, ans in enumerate(self.answers) if ans != True]

        # idx = []
        # for i in range(len(self.questions)):
        #     if i < len(pos_idx):
        #         idx.append(pos_idx[i])
        #     if i < len(neg_idx):
        #         idx.append(neg_idx[i])

        # self.questions = [self.questions[i] for i in idx]
        # self.answers   = [self.answers[i] for i in idx]
        # self.passages  = [self.passages[i] for i in idx]

        self.Qtype = 'BoolQ'

        # self.idx = list(range(len(self.questions)))
        # random.seed(SEED)
        # random.shuffle(self.idx)
        # print(self.idx)

    def gen_question(self, num_questions=-1):
        """
        generator function to iterate over all questions in a dataset
        :param num_questions: number of questions to ask
        :return: a tuple of (question, answer, passage)
        """
        # assert num_questions <= len(self.questions)
        if num_questions == -1:
            num_questions_to_use = len(self.questions)
        else:
            num_questions_to_use = min(len(self.questions), num_questions)
        for i in range(num_questions_to_use):
            yield self.questions[i], self.answers[i], self.passages[i]



    def basic_prompt(self, question, context):
        """
        Prompt used for a single model, or round 0 of debate
        :param question: question to ask
        :param context: additional context, should be the passage for BoolQ
        :return: a prompt which is ready to be passed to an LLM
        """
        prompt = (f'You will be given a yes-no question which is based on a passage. You should use the passage to help you answer the question. '
                  f'You should give a an extremely brief justification for your answer, and you must provide a final answer of either Yes or No.'
                  f'\nQuestion: {question}?'
                  f'\nPassage: {context}')

        return prompt

    def targeted_basic_prompt(self, question, context, answer, correct=True):
        """
        Prompt used for a single model, or round 0 of debate
        :param question: question to ask
        :param context: additional context, should be the passage for BoolQ
        :return: a prompt which is ready to be passed to an LLM
        """
        if correct:
            target_answer = "Yes" if answer else "No"
        else:
            target_answer = "No"  if answer else "Yes"
        
        prompt = (f'You will be given a yes-no question which is based on a passage. You should use the passage to help you answer the question with a {target_answer}. '
                f'You should give a an extremely brief justification for your answer of {target_answer}, and you must state that your final answer is {target_answer}.'
                f'\nQuestion: {question}?'
                f'\nPassage: {context}')

        return prompt



    def debate_prompt(self, question, context, responses, summarized=False, critiques=None):
        """
        prompt for debating
        :param question: question
        :param context: passage (in the case of BoolQ)
        :param responses: a list of responses from the previous round (or the corresponding summarization)
        :param summarized: boolean, whether the responses have been summarized into a single response
        :param critiques: list of critiques, can either be None, or a list with len(critiques) = len(responses)
        :return: a prompt which can be passed directly to an LLM
        """
        prompt =  'Several people have provided answers to a yes-no question. '
        prompt += 'Below are their responses:'

        for i, resp in enumerate(responses):
            prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
    
        prompt += '\n\n'
        prompt += (f'You should take these answers into consideration when answering the following yes-no question which is based on a passage. '
                   f'You should give an extremely brief justification for your answer, and you must provide a final answer of either Yes or No.'
                   f'\nQuestion: {question}?'
                   f'\nPassage: {context}')
        return prompt


    def targeted_debate_prompt(self, question, context, responses, answer, summarized=False, critiques=None, correct=True):
        """
        prompt for debating
        :param question: question
        :param context: passage (in the case of BoolQ)
        :param responses: a list of responses from the previous round (or the corresponding summarization)
        :param summarized: boolean, whether the responses have been summarized into a single response
        :param critiques: list of critiques, can either be None, or a list with len(critiques) = len(responses)
        :return: a prompt which can be passed directly to an LLM
        """
        if correct:
            target_answer = "Yes" if answer else "No"
        else:
            target_answer = "No"  if answer else "Yes"
        
        prompt =  'Several people have provided answers to a yes-no question. '
        prompt += 'Blow are their responses:'

        for i, resp in enumerate(responses):
            prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
    
        prompt += '\n\n'
        prompt += (f'You should take these answers and the passage into consideration when answering the following question with {target_answer}. '
                   f'You should give an extremely brief justification for your answer of {target_answer}, and you must state that your final answer is {target_answer}.'
                   f'\nQuestion: {question}?'
                   f'\nPassage: {context}')
        return prompt


    # single agent detial
    def detail_prompt(self, question, responses, context):
        prompt = 'I am answering a question based on a passage. I would like you to help me improve the correctness of my response by very briefly providing some additional details my original response may have missed. You should be very terse.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        
        prompt += f'\nMy Response: {self.reg.process(responses[0])}'
        return prompt


    def targeted_detail_prompt(self, question, responses, context, answer, correct=True, strategy=0):
        if correct:
            target_answer = "Yes" if answer else "No"
        else:
            target_answer = "No"  if answer else "Yes"
        
        prompt =  f'I would like you to be a debate assistant. You will be given a question based on a passage and my response to the question. You should use the passage and my response to develop two additional details for why the correct answer is {target_answer}. Your details must be extremely brief and must support the fact that the correct answer is {target_answer}.'
        
        if    strategy in [0]:
            prompt += ''
        elif strategy in [1]:            
            prompt += ' Your details should refute my response directly.'
        elif strategy in [2]:
            prompt += ' Your details should support my response.'
        elif strategy in [3]:
            prompt += ' You should be extremely persuasive.'
        elif strategy in [4]:
            prompt += ' Your details should reference both the question and the passage.'
        elif strategy in [5, 6]:
            prompt += ' You should base your details on any expert knowledge that you have.'
        elif strategy in [7, 8]:
            prompt += ' Your details must be based on the passage.'
        elif strategy in [9, 10]:
            prompt += ' Your details should address any misunderstandings in my response.'
        elif strategy in [11, 12]:
            prompt += ' Try not to be too persuasive. '
        elif strategy in [13, 14]:
            prompt += ' It is extremely important that your details are as convincing as possible.'
        elif strategy in [18, 19]:
            prompt += ' You should base your details on general knowledge of the world.'
        

        
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        prompt += f'\nMy Response: {self.reg.process(responses[0])}'
        return prompt

    def summarize_prompt(self, responses, question):
        """
        prompt for summarizing responses between rounds of debate
        when debating with many models, summarization can be effective in terms of both speed and efficacy
        :param responses: list of responses at the previous timestep
        :param question: question (can be helpful in anchoring the summarization)
        :return: a prompt for an LLM to perform summarization
        """
        prompt = f'Several people have given answers to the question \"{question}?\". I want you to provide a summarization of these answers. Your summary should preserve the important components of each response.'
        for i, resp in enumerate(responses):
            prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        return prompt


    def misunderstanding_prompt(self, question, context=None):
        prompt = (
            'I would like you to provide some additional information about a question-answering task, where another model must answer a question based on a given passage. '
            'The model is not allowed to use external resources and must base their answer solely off of the passage and their general understanding of the world. '
            'I would like you to identify places where a model is likely to make an error or have a misunderstanding about this task. '
            'You should also provide additional information to help the other model avoid such errors and misunderstandings. '
            'Below are the question and passage on which you should perform this task.'
            f'\nQuestion: {question}?'
            f'\nPassage: {context}'
            f'\n\nPlease avoid directly answering the question, you should only identify places where a model is likely to make an error and provide additional information to help avoid such mistakes.')
        return prompt

    def misunderstanding_prompt_2(self, paragraph):
        prompt = (
            'Below is a paragraph outlining possible errors and misunderstandings that may occur in a question-answer task, along with proposals on how to avoid those errors. '
            'I would like you to summarize this paragraph while preserving the important elements of the paragraph.'
            f'\nParagraph: {self.reg.process(paragraph)}')
        return prompt

    def devils_advocate_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a question which is based on a passage. I would like you to serve as the devil\'s advocate and provide counter arguments against these answers. You should support your counter arguments by referencing the passage.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt



    def judge_prompt(self, question, responses, context):
        prompt = 'I am answering a question based on a passage, and I would like you to serve as a judge of my answer. You should evaluate whether my response accurately answers the question. You should give an extremely brief justification for your evaluation.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        prompt += f'\nMy Response: {self.reg.process(responses[0])}'
        return prompt


    
        
    def context_prompt(self, question, responses, context):
        prompt = 'I would like you to help provide additional information about a question-answering task. The question is based on a passage. Without directly answering the question, you should provide additional details that you think would be helpful in answering the question. Please try to be as detailed as possible.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        return prompt
    
    # single agent question asking
    def question_ask_prompt(self, question, responses, context):
        prompt = 'I would like you to help me improve my response to a question-answering task. You will be given a question, a passage, and my response to that question. You should ask a single brief clarifying question to help me make my response more accurate.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        prompt += f'\nMy Response : {self.reg.process(responses[0])}'
        return prompt

    def question_answer_prompt(self, question, clarifying_questions, context):
        prompt = 'I am trying to answer a question based on a passage, however, I have some clarification questions that I would like you to answer for me. Please provide detailed answers to my clarification questions.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        prompt += f'\nClarification Questions: {self.reg.process(clarifying_questions)}'
        return prompt



    def refute_prompt(self, question, responses, context):
        prompt = 'I would like you to help me improve my response to a question-answering task based on a passage. You should analyze my responses and check for any errors or misunderstandings. You should give an extremely brief outline of any errors or misunderstandings that you find.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nPassage: {context}'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    
    def helper_prompt(self, question, responses, context):
        prompt = ('I would like you to serve as an assistant and help me answer a yes-no question which is based on a passage. '
                  'You should first select a persona, from the list of possible personas, that you think will be the most helpful in improving my answer. '    # will un-hard-code this list later
                  'You should then help me by roleplaying as that persona. '
                  'You do not need to directly answer the question, but you should help guide me to produce a more accurate and correct answer by roleplaying as your selected persona.'
                  f'\nPossible Personas: Judge, Question-Asker, and Detial.'
                  f'\nQuestion: {question}?'
                  f'\nPassage: {context}'
                  f'\nMy Answer: {self.reg.process(responses[0])}'
        )
        return prompt

    def critic_prompt(self, question, response, context=None):
        prompt = (
            'I would like you to evaluate my answer to a question based on a passage. '
            'Please evaluate my answer and identify any places in which my answer is incorrect or inconsistent with passage. '
            'If you identify any inconsistencies, please provide me a short list of specific details and a brief discussion on how the inconsistencies can be fixed. If you believe that my answer is correct, then please respond with \"Your answer is correct\".'
            f'\nQuestion: {question}?'
            f'\nPassage: {context}'
            f'\n\nMy Answer: {self.reg.process(response)}'
        )
        return prompt

    def make_corrections_prompt(self, question, response, critiques, context=None):
        prompt = (
            'I would like you to make corrections to a response. You will be given a yes-no question, a passage related to that question, a response to that question, and a list of possible issues with the response. '
            'I want you to provide a corrected version of the response. You should base this corrected version on the passage and the list of possible issues. You should try and make as few changes as possible to the original response. '
            'You should only provide a corrected version of the response without restating the question or passage. '
            'It is very important that your corrected version still provides a final answer either Yes or No.'
            f'\nQuestion: {question}?'
            f'\nPassage: {context}'
            f'\n\nResponse: {self.reg.process(response)}'
            f'\n\nPossible Issues: {self.reg.process(critiques)}'
            f'\n\nYour corrected version of the response must still give a conclusive final answer of either Yes or No to the question. You must state this by saying Final Answer.')
        return prompt

    def eval_prompt(self, question, response, critiques, context=None):
        prompt = (
            'Please evaluate whether the following answer correctly answers the following yes-no question. The question is based on information in a passage. Please only respond with \”correct\” or \”incorrect\”. '
            f'\nQuestion: {question}?'
            f'\nPassage: {context}'
            f'\nAnswer: {self.reg.process(response)}'
            )
        return prompt
        
    def test_persona_prompt(self, question, responses, context=None):
        prompt = ('I would like you to serve as a helpful assistant. You should check my answer and decide whether or not it correctly answers the following question based on the passage.'
                 f'\nQuestion: {question}?'
                 f'\nPassage: {context}'
                 f'\nMy Answer: {self.reg.process(responses[0])}')
        return prompt
    


class MMLUQuestion(Question):

    def __init__(self, hold_out_subjects=None, hold_in_subjects=None, verbose=True):
        super().__init__()

        self.cata_by_name = {'abstract_algebra': 'math',
                            'accounting': 'math',
                            'anatomy': 'bio',
                            'astronomy': 'sci',
                            'biology': 'bio',
                            'business_ethics': 'law',
                            'chemistry': 'sci',
                            'clinical_knowledge': 'bio',
                            'computer_science': 'sci',
                            'computer_security': 'sci',
                            'conceptual_physics': 'sci',
                            'econometrics': 'math',
                            'electrical_engineering': 'sci',
                            'elementary_mathematics': 'math',
                            'foreign_policy': 'law',
                            'formal_logic': 'math',
                            'geography': 'gen',
                            'global_facts': 'gen',
                            'government_and_politics': 'law',
                            'history': 'gen',
                            'human_aging': 'bio',
                            'human_sexuality': 'bio',
                            'international_law': 'law',
                            'jurisprudence': 'law',
                            'law': 'law',
                            'logical_fallacies': 'math',
                            'machine_learning': 'sci',
                            'macroeconomics': 'math',
                            'management': 'hum',
                            'marketing': 'hum',
                            'mathematics': 'math',
                            'medical_genetics': 'bio',
                            'medicine': 'bio',
                            'microeconomics': 'math',
                            'miscellaneous': 'gen',
                            'moral_disputes': 'hum',
                            'moral_scenarios': 'hum',
                            'nutrition': 'bio',
                            'philosophy': 'hum',
                            'physics': 'sci',
                            'prehistory': 'gen',
                            'psychology': 'soc_sci',
                            'public_relations': 'hum',
                            'religions': 'soc_sci',
                            'security_studies': 'sci',
                            'sociology': 'soc_sci',
                            'statistics': 'math',
                            'virology': 'bio',
                            }
        self.file_list=[
                        "MMLU/dev/moral_scenarios_dev.csv",
                        "MMLU/dev/high_school_statistics_dev.csv",
                        "MMLU/dev/high_school_macroeconomics_dev.csv",
                        "MMLU/dev/philosophy_dev.csv",
                        "MMLU/dev/world_religions_dev.csv",
                        "MMLU/dev/elementary_mathematics_dev.csv",
                        "MMLU/dev/college_chemistry_dev.csv",
                        "MMLU/dev/high_school_us_history_dev.csv",
                        "MMLU/dev/human_sexuality_dev.csv",
                        "MMLU/dev/professional_psychology_dev.csv",
                        "MMLU/dev/college_computer_science_dev.csv",
                        "MMLU/dev/human_aging_dev.csv",
                        "MMLU/dev/econometrics_dev.csv",
                        "MMLU/dev/conceptual_physics_dev.csv",
                        "MMLU/dev/high_school_mathematics_dev.csv",
                        "MMLU/dev/high_school_psychology_dev.csv",
                        "MMLU/dev/us_foreign_policy_dev.csv",
                        "MMLU/dev/management_dev.csv",
                        "MMLU/dev/high_school_physics_dev.csv",
                        "MMLU/dev/virology_dev.csv",
                        "MMLU/dev/global_facts_dev.csv",
                        "MMLU/dev/astronomy_dev.csv",
                        "MMLU/dev/nutrition_dev.csv",
                        "MMLU/dev/logical_fallacies_dev.csv",
                        "MMLU/dev/electrical_engineering_dev.csv",
                        "MMLU/dev/high_school_computer_science_dev.csv",
                        "MMLU/dev/abstract_algebra_dev.csv",
                        "MMLU/dev/high_school_biology_dev.csv",
                        "MMLU/dev/professional_medicine_dev.csv",
                        "MMLU/dev/anatomy_dev.csv",
                        "MMLU/dev/prehistory_dev.csv",
                        "MMLU/dev/sociology_dev.csv",
                        "MMLU/dev/computer_security_dev.csv",
                        "MMLU/dev/professional_law_dev.csv",
                        "MMLU/dev/college_biology_dev.csv",
                        "MMLU/dev/clinical_knowledge_dev.csv",
                        "MMLU/dev/medical_genetics_dev.csv",
                        "MMLU/dev/marketing_dev.csv",
                        "MMLU/dev/jurisprudence_dev.csv",
                        "MMLU/dev/high_school_european_history_dev.csv",
                        "MMLU/dev/high_school_world_history_dev.csv",
                        "MMLU/dev/college_medicine_dev.csv",
                        "MMLU/dev/high_school_geography_dev.csv",
                        "MMLU/dev/high_school_government_and_politics_dev.csv",
                        "MMLU/dev/security_studies_dev.csv",
                        "MMLU/dev/moral_disputes_dev.csv",
                        "MMLU/dev/high_school_chemistry_dev.csv",
                        "MMLU/dev/college_mathematics_dev.csv",
                        "MMLU/dev/professional_accounting_dev.csv",
                        "MMLU/dev/college_physics_dev.csv",
                        "MMLU/dev/formal_logic_dev.csv",
                        "MMLU/dev/high_school_microeconomics_dev.csv",
                        "MMLU/dev/business_ethics_dev.csv",
                        "MMLU/dev/international_law_dev.csv",
                        "MMLU/dev/miscellaneous_dev.csv",
                        "MMLU/dev/public_relations_dev.csv",
                        "MMLU/dev/machine_learning_dev.csv",
                        "MMLU/val/college_mathematics_val.csv",
                        "MMLU/val/medical_genetics_val.csv",
                        "MMLU/val/machine_learning_val.csv",
                        "MMLU/val/human_aging_val.csv",
                        "MMLU/val/anatomy_val.csv",
                        "MMLU/val/professional_medicine_val.csv",
                        "MMLU/val/conceptual_physics_val.csv",
                        "MMLU/val/high_school_european_history_val.csv",
                        "MMLU/val/world_religions_val.csv",
                        "MMLU/val/professional_psychology_val.csv",
                        "MMLU/val/nutrition_val.csv",
                        "MMLU/val/management_val.csv",
                        "MMLU/val/logical_fallacies_val.csv",
                        "MMLU/val/high_school_computer_science_val.csv",
                        "MMLU/val/high_school_psychology_val.csv",
                        "MMLU/val/prehistory_val.csv",
                        "MMLU/val/formal_logic_val.csv",
                        "MMLU/val/astronomy_val.csv",
                        "MMLU/val/college_biology_val.csv",
                        "MMLU/val/high_school_macroeconomics_val.csv",
                        "MMLU/val/security_studies_val.csv",
                        "MMLU/val/high_school_statistics_val.csv",
                        "MMLU/val/high_school_mathematics_val.csv",
                        "MMLU/val/high_school_physics_val.csv",
                        "MMLU/val/high_school_chemistry_val.csv",
                        "MMLU/val/high_school_us_history_val.csv",
                        "MMLU/val/high_school_world_history_val.csv",
                        "MMLU/val/college_computer_science_val.csv",
                        "MMLU/val/jurisprudence_val.csv",
                        "MMLU/val/virology_val.csv",
                        "MMLU/val/public_relations_val.csv",
                        "MMLU/val/college_chemistry_val.csv",
                        "MMLU/val/us_foreign_policy_val.csv",
                        "MMLU/val/college_physics_val.csv",
                        "MMLU/val/high_school_microeconomics_val.csv",
                        "MMLU/val/college_medicine_val.csv",
                        "MMLU/val/moral_disputes_val.csv",
                        "MMLU/val/philosophy_val.csv",
                        "MMLU/val/elementary_mathematics_val.csv",
                        "MMLU/val/electrical_engineering_val.csv",
                        "MMLU/val/marketing_val.csv",
                        "MMLU/val/high_school_government_and_politics_val.csv",
                        "MMLU/val/sociology_val.csv",
                        "MMLU/val/international_law_val.csv",
                        "MMLU/val/abstract_algebra_val.csv",
                        "MMLU/val/professional_law_val.csv",
                        "MMLU/val/high_school_biology_val.csv",
                        "MMLU/val/professional_accounting_val.csv",
                        "MMLU/val/clinical_knowledge_val.csv",
                        "MMLU/val/computer_security_val.csv",
                        "MMLU/val/high_school_geography_val.csv",
                        "MMLU/val/global_facts_val.csv",
                        "MMLU/val/econometrics_val.csv",
                        "MMLU/val/miscellaneous_val.csv",
                        "MMLU/val/business_ethics_val.csv",
                        "MMLU/val/moral_scenarios_val.csv",
                        "MMLU/val/human_sexuality_val.csv",
                        "MMLU/test/professional_psychology_test.csv",
                        "MMLU/test/sociology_test.csv",
                        "MMLU/test/anatomy_test.csv",
                        "MMLU/test/professional_law_test.csv",
                        "MMLU/test/college_biology_test.csv",
                        "MMLU/test/prehistory_test.csv",
                        "MMLU/test/moral_disputes_test.csv",
                        "MMLU/test/econometrics_test.csv",
                        "MMLU/test/conceptual_physics_test.csv",
                        "MMLU/test/elementary_mathematics_test.csv",
                        "MMLU/test/college_computer_science_test.csv",
                        "MMLU/test/high_school_world_history_test.csv",
                        "MMLU/test/high_school_geography_test.csv",
                        "MMLU/test/professional_accounting_test.csv",
                        "MMLU/test/high_school_mathematics_test.csv",
                        "MMLU/test/international_law_test.csv",
                        "MMLU/test/business_ethics_test.csv",
                        "MMLU/test/high_school_chemistry_test.csv",
                        "MMLU/test/high_school_statistics_test.csv",
                        "MMLU/test/college_chemistry_test.csv",
                        "MMLU/test/jurisprudence_test.csv",
                        "MMLU/test/astronomy_test.csv",
                        "MMLU/test/high_school_macroeconomics_test.csv",
                        "MMLU/test/miscellaneous_test.csv",
                        "MMLU/test/computer_security_test.csv",
                        "MMLU/test/marketing_test.csv",
                        "MMLU/test/high_school_biology_test.csv",
                        "MMLU/test/virology_test.csv",
                        "MMLU/test/college_physics_test.csv",
                        "MMLU/test/management_test.csv",
                        "MMLU/test/world_religions_test.csv",
                        "MMLU/test/security_studies_test.csv",
                        "MMLU/test/medical_genetics_test.csv",
                        "MMLU/test/electrical_engineering_test.csv",
                        "MMLU/test/logical_fallacies_test.csv",
                        "MMLU/test/clinical_knowledge_test.csv",
                        "MMLU/test/abstract_algebra_test.csv",
                        "MMLU/test/human_aging_test.csv",
                        "MMLU/test/high_school_government_and_politics_test.csv",
                        "MMLU/test/philosophy_test.csv",
                        "MMLU/test/high_school_us_history_test.csv",
                        "MMLU/test/high_school_physics_test.csv",
                        "MMLU/test/high_school_computer_science_test.csv",
                        "MMLU/test/high_school_psychology_test.csv",
                        "MMLU/test/human_sexuality_test.csv",
                        "MMLU/test/professional_medicine_test.csv",
                        "MMLU/test/nutrition_test.csv",
                        "MMLU/test/college_mathematics_test.csv",
                        "MMLU/test/high_school_microeconomics_test.csv",
                        "MMLU/test/global_facts_test.csv",
                        "MMLU/test/formal_logic_test.csv",
                        "MMLU/test/machine_learning_test.csv",
                        "MMLU/test/us_foreign_policy_test.csv",
                        "MMLU/test/public_relations_test.csv",
                        "MMLU/test/high_school_european_history_test.csv",
                        "MMLU/test/college_medicine_test.csv",
                        "MMLU/test/moral_scenarios_test.csv",
                        ]

        if hold_out_subjects is not None and hold_in_subjects is not None:
            print(f"either hold_out_subjects and hold_in_subjects must be None, but you have\nhold_out_subjects={hold_out_subjects}\nhold_in_subjects={hold_in_subjects}")
            exit(-1)
        
        self.hold_out_subjects = hold_out_subjects
        self.hold_in_subjects = hold_in_subjects

        directory_list = ['MMLU/dev/', 'MMLU/val/', 'MMLU/test/']
        self.catas = []
        df = pd.DataFrame(columns=['Question', 'A', 'B', 'C', 'D', 'Answer'])
        # for directory in directory_list:
        #     for file in os.listdir(directory):
        #         if file.endswith(".csv"):
        for file_path in self.file_list:
            new_df = pd.read_csv(file_path)
            new_df.columns = ['question', 'A', 'B', 'C', 'D', 'answer']
            df = pd.concat([df, new_df], ignore_index=True)

            for sub_name, cata in self.cata_by_name.items():
                if sub_name in file_path:
                    self.catas += [cata]*new_df.shape[0]
                    break

        self.letter_options = [f'({elem})' for elem in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
        self.Qtype = 'MMLUQ'

        self.questions = df['question']
        self.answers = df['answer']
        self.options = [(a, b, c, d) for a, b, c, d in
                        zip(df['A'].to_list(), df['B'].to_list(), df['C'].to_list(), df['D'].to_list())]
        if verbose:
            print(f'Num Questions: {len(self.options)}')
            print(f'Num Questions: {len(self.catas)}')

        

    def gen_question(self, num_questions=-1):
        if num_questions == -1:
            n = len(self.questions)
        else:
            n = min(len(self.questions), num_questions)
        for i in range(n):
            yield self.questions[i], self.answers[i], self.options[i]


    def basic_prompt(self, question, context):

        prompt = (f'Please answer the following multiple choice question as accurately as possible. '
                #   f'You must provide a extremely brief justification for your answer, and you must give a final answer of either (A), (B), (C), or (D), by saying \"Final Answer:\".'
                f'You must provide a extremely brief justification for your answer, and you must give your final answer as a letter by saying \"Final Answer:\".'
                  f'\nQuestion: {question},')
        prompt += '\nOptions:'
        for letter, option in zip(self.letter_options, context):
            prompt += f'\n{letter}: {option}'
        return prompt
    
    def targeted_basic_prompt(self, question, context, answer, correct=True):
        target_answer = answer if correct else random.choice([elem for elem, _ in zip(self.letter_options, context) if elem!=answer])
        prompt = (f'Please answer the following multiple choice question with option {target_answer}. '
                  f'You must provide a extremely brief justification for your answer of {target_answer}, and you must give a final answer of {target_answer}, by saying \"Final Answer: {target_answer}\".'
                  f'\nQuestion: {question},')
        prompt += '\nOptions:'
        for letter, option in zip(self.letter_options, context):
            prompt += f'\n{letter}: {option}'

        return prompt

    def debate_prompt(self, question, responses, context, critiques=None):
        prompt = 'Several people have provided answers to a multiple choice question. Below are their responses:'
        for i, resp in enumerate(responses):
            prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        prompt += '\n\n'
        prompt += (
                f'You should take these answers into consideration when answering the following multiple choice question. '
                # f'You must give an extremely brief justification for your answer, and you must provide a final answer of either (A), (B), (C), or (D), by saying \"Final Answer:\".'
                f'You must give an extremely brief justification for your answer, and you must provide your final answer as a letter by saying \"Final Answer:\".'
                f'\nQuestion: {question}')
        prompt += '\nOptions:'
        for letter, option in zip(self.letter_options, context):
            prompt += f'\n{letter}: {option}'
        return prompt

    def targeted_debate_prompt(self, question, responses, context, answer, correct=1):
        target_answer = answer if correct else random.choice([elem for elem, _ in zip(self.letter_options, context) if elem!=answer])
        prompt = 'Several people have provided answers to a multiple choice question. Below are their responses:'
        for i, resp in enumerate(responses):
            prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        prompt += '\n\n'
        prompt += (
                f'You should take these answers into consideration and answer the following multiple choice question with option {target_answer}. '
                f'You must give an extremely brief justification for your answer, and you must provide a final answer of {target_answer} by saying \"Final Answer: {target_answer}\".'
                f'\nQuestion: {question}')
        prompt += '\nOptions:'
        for letter, option in zip(self.letter_options, context):
            prompt += f'\n{letter}: {option}'
        return prompt

    def detail_prompt(self, question, responses, context):
        prompt = 'I am answering a multiple choice question. I would like you to help me improve the correctness of my response by very briefly providing some additional details my original response may have missed. You should be very terse.'
        prompt += f'\nQuestion: {question}?'
        prompt += '\nOptions:'
        for letter, option in zip(self.letter_options, context):
            prompt += f'\n{letter}: {option}'
        prompt += f'\nMy Response: {self.reg.process(responses[0])}'
        return prompt

    def targeted_detail_prompt(self, question, responses, context, answer, correct=True, strategy=0):
        target_answer = answer if correct else random.choice([elem for elem, _ in zip(self.letter_options, context) if elem!=answer])
        prompt = 'I would like you to be a debate assistant. You will be given a multiple choice question, '
        
        prompt += f'you should provide two additional details for why the correct answer is {target_answer}. Your details must be extremely brief and must support the fact that the correct answer is {target_answer}.'
        
        if    strategy in [0]:
            prompt += ''
        elif strategy in [1]:            
            prompt += ' Your details should refute my response directly.'
        elif strategy in [2]:
            prompt += ' Your details should support my response.'
        elif strategy in [3]:
            prompt += ' You should be extremely persuasive. '
        elif strategy in [4]:
            prompt += ' Your details should reference both the question and multiple choice options.'
        elif strategy in [5, 6]:
            prompt += ' You should base your details on any expert knowledge that you have.'
        elif strategy in [7, 8]:
            prompt += ' Your details must be based on the multiple choice options.'
        elif strategy in [9, 10]:
            prompt += ' Your details should address any misunderstandings in my response.'
        elif strategy in [11, 12]:
            prompt += ' Try not to be too persuasive. '
        elif strategy in [13, 14]:
            prompt += ' It is extremely important that your details are as convincing as possible.'
        elif strategy in [18, 19]:
            prompt += ' You should base your details on general knowledge of the world.'
        
        prompt += f'\nQuestion: {question}?'
        prompt += '\nOptions:'
        for letter, option in zip(self.letter_options, context):
            prompt += f'\n{letter}: {option}'
        prompt += f'\nMy Response: {self.reg.process(responses[0])}'
        return prompt

    def devils_advocate_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a multiple choice question. I would like you to serve as the devil\'s advocate and provide counter arguments against these answers. You should support your counter arguments with factual information and logical reasoning.'
        prompt += f'\nQuestion: {question}'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def judge_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a multiple choice question. I would like you to serve as a judge of these answers. You should evaluate whether each answer is correct. You should support your evaluation with factual information and logical reasoning.'
        prompt += f'\nQuestion: {question}'

        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def context_prompt(self, question, responses, context):
        prompt = 'I would like you to help provide additional information about a multiple choice question. Without directly answering the question, you should provide additional details that you think would be helpful in answering the question. Please try to be as detailed as possible and support these detials with factual information and logical reasoning.'
        prompt += f'\nQuestion: {question}'
        return prompt

    def question_ask_prompt(self, question, responses, context):
        prompt = 'I would like you to help me improve a multiple choice question-answering task. You will be given the question and a list of responses to the question. You should ask clarifying questions about any part of the question or responses that you believe requires additional information or is not grounded in factual information and logical reasoning.'
        prompt += f'\nQuestion: {question}'
        for i, response in enumerate(responses):
            prompt += f'\nResponse {i}: {self.reg.process(response)}'
        return prompt

    def question_answer_prompt(self, question, clarifying_questions, context):
        prompt = 'I am trying to answer a multiple choice question. However, I have some clarification questions that I would like you to answer for me. Please provide detailed answers to my clarification questions.'
        prompt += f'\nQuestion: {question}'
        prompt += f'\nClarification Questions: {self.reg.process(clarifying_questions)}'
        return prompt

    def refute_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a multiple choice question. I would like you to analyze these responses and check for any errors, misunderstandings. Please clearly outline any errors or misunderstandings that you find. Additionally, you should provide details on how these errors and misunderstandings can be fixed. You should support your arguments with factual information and logical reasoning.'
        prompt += f'\nQuestion: {question}'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def helper_prompt(self, question, responses, context):
        prompt = ('I would like you to serve as an assistant and help me answer a multiple choice question. '
                  'I want you to help me by discussing the question and my answer. '
                  'You do not need to directly answer the question, but you should help guide me to produce a more accurate and correct answer.'
                  f'\nQuestion: {question}'
                  f'\nMy Answer: {self.reg.process(responses[0])}'
        )
        return prompt





class BBHQuestion(Question):


    def __init__(self, hold_out_subjects=None, hold_in_subjects=None):
        super().__init__()
        
        self.Qtype = 'BBH'
        self.MMLUQuestion = MMLUQuestion(verbose=False)
        
        self.multiple_choice_files = ['temporal_sequences.json', 
                                    'logical_deduction_three_objects.json', 
                                    'date_understanding.json', 
                                    'hyperbaton.json', 
                                    'penguins_in_a_table.json', 
                                    'reasoning_about_colored_objects.json', 
                                    'snarks.json', 
                                    'tracking_shuffled_objects_seven_objects.json', 
                                    'movie_recommendation.json', 
                                    'logical_deduction_five_objects.json', 
                                    'logical_deduction_seven_objects.json', 
                                    'geometric_shapes.json', 
                                    'disambiguation_qa.json', 
                                    'tracking_shuffled_objects_five_objects.json', 
                                    'ruin_names.json', 
                                    'tracking_shuffled_objects_three_objects.json', 
                                    'salient_translation_error_detection.json']

        self.yes_no_files          = ['navigate.json', 
                                    'causal_judgement.json', 
                                    'boolean_expressions.json', 
                                    'sports_understanding.json', 
                                    'web_of_lies.json']


        self.subjects = {'temporal_sequences.json':  'logic', 
                        'navigate.json': 'reasoning', 
                        'causal_judgement.json': 'logic', 
                        'logical_deduction_three_objects.json': 'logic',
                        'date_understanding.json': 'reasoning',
                        'hyperbaton.json': 'language', 
                        'penguins_in_a_table.json': 'reasoning', 
                        'reasoning_about_colored_objects.json': 'reasoning', 
                        'snarks.json': 'language',
                        'tracking_shuffled_objects_seven_objects.json': 'reasoning',
                        'movie_recommendation.json': 'language',
                        'logical_deduction_five_objects.json': 'logic',
                        'logical_deduction_seven_objects.json': 'logic',
                        'geometric_shapes.json': 'reasoning',
                        'disambiguation_qa.json': 'language',
                        'sports_understanding.json': 'language',
                        'tracking_shuffled_objects_five_objects.json': 'reasoning',
                        'web_of_lies.json': 'logic',
                        'ruin_names.json': 'language',
                        'tracking_shuffled_objects_three_objects.json': 'reasoning',
                        'salient_translation_error_detection.json': 'language',
                        'boolean_expressions.json': 'logic'}
        
        self.questions      = []
        self.options        = []
        self.answers        = []
        self.question_types = []
        self.catas          = []

        self.hold_in_subjects  = hold_in_subjects
        self.hold_out_subjects = hold_out_subjects

        path = 'BBH/'
        tr_idx = []
        ts_idx = []
        q_idx = 0
        # for file_name in os.listdir(path):
        for file_name in self.multiple_choice_files + self.yes_no_files:
            if 'json' in file_name:
                with open(path+file_name, 'r', encoding='utf-8') as file:
                    data = [json.loads(line) for line in file if line.strip()]
                    
                    if file_name in self.yes_no_files:
                        for elem in data[0]['examples']:
                            if q_idx%5==1:
                                ts_idx.append(q_idx)
                            else:
                                tr_idx.append(q_idx)
                            q_idx += 1
                            self.catas.append(self.subjects[file_name])
                            self.options.append(None)
                            question  = elem['input']
                            answer    = elem['target']

                            # web_of_lies has a unique format
                            if file_name == 'web_of_lies.json':
                                self.questions.append(question.replace('Question: ', ''))
                            elif file_name == 'boolean_expressions.json':
                                self.questions.append(f'Is the following boolean expression True? {question[:-3]}')
                            else:
                                self.questions.append(question)

                            if   answer.lower() == 'yes':
                                self.answers.append(1)
                            elif answer.lower() == 'no':
                                self.answers.append(0)
                            elif file_name == 'boolean_expressions.json':
                                self.answers.append(int(answer=='True'))
                            else:
                                print(file_name)
                                print(answer)
                                print(question)
                                print("ERROR IN BBH YES-NO QUESTION")

                            # no options for yes-no questions
                    
                    elif file_name in self.multiple_choice_files:
                        for elem in data[0]['examples']:
                            if q_idx%5==1:
                                ts_idx.append(q_idx)
                            else:
                                tr_idx.append(q_idx)
                            q_idx += 1

                            self.catas.append(self.subjects[file_name])
                            question  = elem['input']
                            answer    = elem['target']

                            question, options = question.split('\nOptions:\n')
                            actual_options = [option[4:] for option in options.split('\n')]

                            self.questions.append(question)
                            self.answers.append(answer.replace('(', '').replace(')', '').upper())
                            self.options.append(actual_options)

        self.questions = [self.questions[i] for i in tr_idx] + [self.questions[i] for i in ts_idx]
        self.answers  = [self.answers[i]    for i in tr_idx] + [self.answers[i]   for i in ts_idx]
        self.options  = [self.options[i]    for i in tr_idx] + [self.options[i]   for i in ts_idx]
        self.catas    = [self.catas[i]      for i in tr_idx] + [self.catas[i]     for i in ts_idx]

    def gen_question(self, num_questions=-1):
        # assert num_questions <= len(self.questions)
        if num_questions == -1:
            num_questions_to_use = len(self.questions)
        else:
            num_questions_to_use = min(len(self.questions), num_questions)
        for i in range(num_questions_to_use):
            yield self.questions[i], self.answers[i], self.options[i]



    def basic_prompt(self, question, context):

        # context != None => multiple choice
        if context is not None: 
            prompt = self.MMLUQuestion.basic_prompt(question, context)
        else:
            prompt = (f'You will be given a yes-no question. You should answer the question as accurately as possible. '
                      f'You should give a an extremely brief justification for your answer, and you must provide a final answer of either Yes or No.'
                      f'\nQuestion: {question}')

        return prompt

    def targeted_basic_prompt(self, question, context, answer, correct=True):
        if context is not None:
            prompt = self.MMLUQuestion.targeted_basic_prompt(question=question,
                                                             context=context,
                                                             answer=answer,
                                                             correct=correct)
        else:
            if correct:
                target_answer = "Yes" if answer else "No"
            else:
                target_answer = "No"  if answer else "Yes"
            
            prompt = (f'You will be given a yes-no question. You should answer the question with {target_answer}. '
                      f'You should give a an extremely brief justification for your answer of {target_answer}, and you must state that your final answer is {target_answer}.'
                      f'\nQuestion: {question}')

        return prompt


    def debate_prompt(self, question, context, responses, critiques=None):

        if context is not None:
            prompt = self.MMLUQuestion.debate_prompt(question=question,
                                                     context=context,
                                                     responses=responses,
                                                     critiques=critiques)
        else:
            prompt =  'Several people have provided answers to a yes-no question. '
            prompt += 'Blow are their responses:'

            for i, resp in enumerate(responses):
                prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        
            prompt += '\n\n'
            prompt += (f'You should take these answers into consideration when answering the following yes-no question. '
                       f'You should give an extremely brief justification for your answer, and you must provide a final answer of either Yes or No.'
                       f'\nQuestion: {question}')
        return prompt


    def targeted_debate_prompt(self, question, context, responses, answer, correct=True):
        if context is not None:
            prompt = self.MMLUQuestion.targeted_debate_prompt(question=question,
                                                              context=context,
                                                              responses=responses,
                                                              answer=answer,
                                                              correct=correct)
        else:
            if correct:
                target_answer = "Yes" if answer else "No"
            else:
                target_answer = "No"  if answer else "Yes"
            
            prompt =  'Several people have provided answers to a yes-no question. '
            prompt += 'Blow are their responses:'

            for i, resp in enumerate(responses):
                prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        
            prompt += '\n\n'
            prompt += (f'You should take these answers and the passage into consideration when answering the following question with {target_answer}. '
                       f'You should give an extremely brief justification for your answer of {target_answer}, and you must state that your final answer is {target_answer}.'
                       f'\nQuestion: {question}')
        return prompt


    # single agent detail
    def detail_prompt(self, question, responses, context):
        if context is not None:
            prompt = self.MMLUQuestion.detail_prompt(question=question, responses=responses, context=context)
        else:
            prompt  = 'I am answering a yes-no question. I would like you to help me improve the correctness of my response by very briefly providing some additional details my original response may have missed. You should be very terse.'
            prompt += f'\nQuestion: {question}'
            
            prompt += f'\nMy Response: {self.reg.process(responses[0])}'
        return prompt


    def targeted_detail_prompt(self, question, responses, context, answer, correct=True, strategy=0):
        if context is not None:
            prompt = self.MMLUQuestion.targeted_detail_prompt(question=question,
                                                              responses=responses,
                                                              context=context,
                                                              answer=answer,
                                                              correct=correct,
                                                              strategy=strategy)
        else:
            if correct:
                target_answer = "Yes" if answer else "No"
            else:
                target_answer = "No"  if answer else "Yes"
            
            prompt =  f'I would like you to be a debate assistant. You will be given a yes-no question and my response to the question. You should use the question and my response to develop two additional details for why the correct answer is {target_answer}. Your details must be extremely brief and must support the fact that the correct answer is {target_answer}.'
            
            if    strategy in [0]:
                prompt += ''
            elif strategy in [1]:            
                prompt += ' Your details should refute my response directly.'
            elif strategy in [2]:
                prompt += ' Your details should support my response.'
            elif strategy in [3]:
                prompt += ' You should be extremely persuasive. '
            elif strategy in [4]:
                prompt += ' Your details should reference both the question and the passage.'
            elif strategy in [5, 6]:
                prompt += ' You should base your details on any expert knowledge that you have.'
            elif strategy in [7, 8]:
                prompt += ' Your details must be based on the passage.'
            elif strategy in [9, 10]:
                prompt += ' Your details should address any misunderstandings in my response.'
            elif strategy in [11, 12]:
                prompt += ' Try not to be too persuasive. '
            elif strategy in [13, 14]:
                prompt += ' It is extremely important that your details are as convincing as possible.'
            elif strategy in [18, 19]:
                prompt += ' You should base your details on general knowledge of the world.'
            

            
            prompt += f'\nQuestion: {question}'
            prompt += f'\nMy Response: {self.reg.process(responses[0])}'
        return prompt




class SCIQQuestion(Question):
    
    def __init__(self, hold_out_subjects=None, hold_in_subjects=None):
        super().__init__()
        
        self.hold_out_subjects = hold_out_subjects
        self.hold_in_subjects  = hold_in_subjects
        
        path = 'SCIQ/'
        self.Qtype = 'SCIQ'
        files = ['train-00000-of-00001.parquet', 'validation-00000-of-00001.parquet', 'test-00000-of-00001.parquet']
        df = pd.DataFrame()
        for file in files:
            new_df = pd.read_parquet(path+file)
            df     = pd.concat([df, new_df], ignore_index=True)
        self.df = df
        self.questions = self.df['question'].to_list()
        self.all_options = self.df[['distractor3', 'distractor1', 'distractor2', 'correct_answer']].to_numpy()
        self.idx = np.zeros((self.df.shape[0], 4), dtype=int)
        for i in range(self.idx.shape[0]):
            self.idx[i] = [i%4, (i+1)%4, (i+2)%4, (i+3)%4]    
        letter_options  = ['A', 'B', 'C', 'D']
        self.answers = [letter_options[(3-i)%4] for i in range(self.idx.shape[0])]
        self.options = [[self.all_options[i][j] for j in self.idx[i]] for i in range(len(self.idx))]

        self.MMLUQuestion = MMLUQuestion(verbose=False)
    
    def gen_question(self, num_questions=-1):
        if num_questions == -1:
            n = len(self.questions)
        else:
            n = min(len(self.questions), num_questions)
        for i in range(n):
            yield self.questions[i], self.answers[i], self.options[i]


    def basic_prompt(self, question, context):
        prompt = self.MMLUQuestion.basic_prompt(question=question, context=context)
        return prompt
    
    def targeted_basic_prompt(self, question, context, answer, correct=True):
        prompt = self.MMLUQuestion.targeted_basic_prompt(question=question, context=context, answer=answer, correct=correct)
        return prompt

    def debate_prompt(self, question, responses, context, critiques=None):
        prompt = self.MMLUQuestion.debate_prompt(question=question, responses=responses, context=context, critiques=critiques)
        return prompt

    def targeted_debate_prompt(self, question, responses, context, answer, correct=1):
        prompt = self.MMLUQuestion.targeted_debate_prompt(question=question, responses=responses, context=context, answer=answer, correct=correct)
        return prompt

    def detail_prompt(self, question, responses, context):
        prompt = self.MMLUQuestion.detail_prompt(question=question, responses=responses, context=context)
        return prompt

    def targeted_detail_prompt(self, question, responses, context, answer, correct=True, strategy=0):
        prompt = self.MMLUQuestion.targeted_detail_prompt(question=question, responses=responses, context=context, answer=answer, correct=correct, strategy=strategy)
        return prompt



class ARCQuestion(Question):
    
    def __init__(self, hold_out_subjects=None, hold_in_subjects=None):
        super().__init__()
        
        self.hold_out_subjects = hold_out_subjects
        self.hold_in_subjects  = hold_in_subjects
        
        path = 'ARC/'
        self.Qtype='ARC'

        # 2 => easy questions
        # 3 => hard questions
        files = ['train-00000-of-00001-2.parquet',       'train-00000-of-00001-3.parquet', 
                 'validation-00000-of-00001-2.parquet',  'validation-00000-of-00001-3.parquet', 
                'test-00000-of-00001-2.parquet',         'test-00000-of-00001-3.parquet']


        df = pd.DataFrame()
        for file in files:
            new_df = pd.read_parquet(path+file)
            df     = pd.concat([df, new_df], ignore_index=True)
        self.df = df
        self.questions = self.df['question'].to_list()
        self.options   = [list(self.df['choices'][ii]['text']) for ii in range(self.df.shape[0])]
        self.answers   = self.df['answerKey'].to_list()
        self.MMLUQuestion = MMLUQuestion(verbose=False)


    def gen_question(self, num_questions=-1):
        if num_questions == -1:
            n = len(self.questions)
        else:
            n = min(len(self.questions), num_questions)
        for i in range(n):
            yield self.questions[i], self.answers[i], self.options[i]


    def basic_prompt(self, question, context):
        prompt = self.MMLUQuestion.basic_prompt(question=question, context=context)
        return prompt
    
    def targeted_basic_prompt(self, question, context, answer, correct=True):
        prompt = self.MMLUQuestion.targeted_basic_prompt(question=question, context=context, answer=answer, correct=correct)
        return prompt

    def debate_prompt(self, question, responses, context, critiques=None):
        prompt = self.MMLUQuestion.debate_prompt(question=question, responses=responses, context=context, critiques=critiques)
        return prompt

    def targeted_debate_prompt(self, question, responses, context, answer, correct=1):
        prompt = self.MMLUQuestion.targeted_debate_prompt(question=question, responses=responses, context=context, answer=answer, correct=correct)
        return prompt

    def detail_prompt(self, question, responses, context):
        prompt = self.MMLUQuestion.detail_prompt(question=question, responses=responses, context=context)
        return prompt

    def targeted_detail_prompt(self, question, responses, context, answer, correct=True, strategy=0):
        prompt = self.MMLUQuestion.targeted_detail_prompt(question=question, responses=responses, context=context, answer=answer, correct=correct, strategy=strategy)
        return prompt




class MedMCQAQuestion(Question):
    
    def __init__(self, hold_out_subjects=None, hold_in_subjects=None):
        super().__init__()
        
        self.hold_out_subjects = hold_out_subjects
        self.hold_in_subjects  = hold_in_subjects
        
        path = 'MedMCQA/'
        self.Qtype = 'MedMCQA'
        # test.csv has no labels
        files = ['train.csv', 'validation.csv']# , 'test.csv'] 
        df = pd.DataFrame()
        for file in files:
            new_df = pd.read_csv(path+file)
            df     = pd.concat([df, new_df], ignore_index=True)
        self.df = df
        self.questions = self.df['question'].to_list()
        self.options   = [list(itm) for itm in self.df[['opa', 'opb', 'opc', 'opd']].to_numpy()]
        self.letter_options = ['A', 'B', 'C', 'D']
        self.answer_idx = self.df['cop'].to_numpy(dtype=int)

        self.answers = [self.letter_options[ii] for ii in self.answer_idx]

        idx = list(range(len(self.answers)))[:182_000]
        np.random.seed(101)
        idx = np.random.choice(idx, size=10_000, replace=False)
        
        self.questions = [self.questions[i] for i in idx] + self.questions[182_000:]
        self.answer    = [self.answers[i] for i in idx]   + self.answers[182_000:]
        self.options   = [self.options[i] for i in idx]   + self.options[182_000:]

        self.MMLUQuestion = MMLUQuestion(verbose=False)
    

    def gen_question(self, num_questions=-1):
        if num_questions == -1:
            n = len(self.questions)
        else:
            n = min(len(self.questions), num_questions)
        for i in range(n):
            yield self.questions[i], self.answers[i], self.options[i]


    def basic_prompt(self, question, context):
        prompt = self.MMLUQuestion.basic_prompt(question=question, context=context)
        return prompt
    
    def targeted_basic_prompt(self, question, context, answer, correct=True):
        prompt = self.MMLUQuestion.targeted_basic_prompt(question=question, context=context, answer=answer, correct=correct)
        return prompt

    def debate_prompt(self, question, responses, context, critiques=None):
        prompt = self.MMLUQuestion.debate_prompt(question=question, responses=responses, context=context, critiques=critiques)
        return prompt

    def targeted_debate_prompt(self, question, responses, context, answer, correct=1):
        prompt = self.MMLUQuestion.targeted_debate_prompt(question=question, responses=responses, context=context, answer=answer, correct=correct)
        return prompt

    def detail_prompt(self, question, responses, context):
        prompt = self.MMLUQuestion.detail_prompt(question=question, responses=responses, context=context)
        return prompt

    def targeted_detail_prompt(self, question, responses, context, answer, correct=True, strategy=0):
        prompt = self.MMLUQuestion.targeted_detail_prompt(question=question, responses=responses, context=context, answer=answer, correct=correct, strategy=strategy)
        return prompt




class ArithmaticQuestion(Question):

    def __init__(self):
        super().__init__()

        self.Qtype = 'MathQ'

        np.random.seed(101)
        self.all_nums = itr.product(np.arange(5)*2 + 3, repeat=6)
        self.all_nums = np.array(list(self.all_nums))
        idx = np.arange(len(self.all_nums))

        np.random.shuffle(idx)

        self.all_nums = self.all_nums[idx]

    def gen_question(self, num_questions=-1):
        if num_questions == -1:
            n = len(self.questions)
        else:
            n = min(len(self.questions), num_questions)
        for i in range(n):
            # a, b, c, d, e, f, = np.random.randint(5, 15, size=6)
            a, b, c, d, e, f = self.all_nums[i]
            question = f'What is the result of the following expression ({a}*{b}*{c}) + ({d}*{e}*{f})'
            answer = a * b * c + d * e * f
            context = None
            yield question, answer, context

    def basic_prompt(self, question, context=None, justification=True):
        prompt = (
            f'{question}? '
            f'You should give a brief justification for your answer. You must provide your final answer as a single number by saying \"Final Answer:\".'
        )
        return prompt

    def summarize_prompt(self, responses, question):
        prompt = (f'Several people have provided several evaluations of the question \"{question}?\". '
                  'I want you to provide a summarization of the given evaluations. '
                  'You should avoid commenting on the correctness and quality of each answer. '
                  'Your summary should preserve the person\’s the important components and additional details provided by the person.')
        for i, resp in enumerate(responses):
            prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        return prompt

    def debate_prompt(self, question, responses, summarized=False, context=None):
        prompt = 'Several people have provided responses to an arithmatic question. '
        if summarized:
            prompt += f'Here is a summary of their responses:\n{self.reg.process(responses)}'
        else:
            prompt += f'Blow are their responses:'
            for i, resp in enumerate(responses):
                prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        prompt += '\n\n'

        prompt += (
            f'You should take these responses into consideration when providing your own answer to the question. '
            f'You should give a very brief justification for your answer. You must provide your final answer as a single number by saying \"Final Answer:\".'
            f'\nQuestion: {question}?')

        return prompt

    def critic_prompt(self, question, response, context=None):
        prompt = (
            'I would like you to be the judge of an answer to arithmatic question. You will be provided with a question, and an answer to that question. '
            'Please evaluate the answer and identify any places in which the given answer is incorrect or inconsistent. '
            'If you identify any inconsistencies, please provide a list of specific details and discuss how the inconsistencies can be fixed.'
            f'\nQuestion: {question}?')

        prompt += f'\n\nGiven Answer: {self.reg.process(response)}'

        return prompt

    def make_corrections_prompt(self, question, response, critiques, context=None):
        prompt = (
            'I would like you to make corrections to a given response. You will be given a question, a response to that question, and a list of possible issues with the given response. '
            'Please provide a corrected version of the given response based on the list of possible issues. You should try and make as few changes as possible to the original answer. '
            'It is very important that your corrected version still provides a final answer as a single number by saying \"Final Answer\".'
            f'\nQuestion:  {question}?')
        prompt += (
            f'\n\nGiven Response: {self.reg.process(response)}'
            f'\n\nPossible Issues: {self.reg.process(critiques)}')
        return prompt

    def devils_advocate_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to an arithmetic question. I would like you to serve as the devil\'s advocate and provide counter arguments against these answers. You should support your counter arguments with accurate mathematical derivations.'
        prompt += f'\nQuestion: {question}?'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def judge_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to an arithmetic question. I would like you to serve as a judge of these answers. You should evaluate whether each answer is correct. You should support your evaluation with mathematical reasoning.'
        prompt += f'\nQuestion: {question}?'

        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def detail_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to an arithmetic question. I would like you to provide additional details that these answers may have missed. You should try to provide as many helpful detail as possible, along with accurate mathematical justification.'
        prompt += f'\nQuestion: {question}?'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def context_prompt(self, question, responses, context):
        prompt = 'I would like you to help provide additional information about an arithmetic question. Without directly answering the question, you should provide additional details that you think would be helpful in answering the question. Please try to be as detailed as possible and as mathematically accurate as possible.'
        prompt += f'\nQuestion: {question}?'
        return prompt

    def question_ask_prompt(self, question, responses, context):
        prompt = 'I would like you to help me improve an arithmetic question-answering task. You will be given the question and a list of responses to the question. You should ask clarifying questions about any part of the question or responses that you believe requires additional information or is not mathematically accurate.'
        prompt += f'\nQuestion: {question}?'
        for i, response in enumerate(responses):
            prompt += f'\nResponse {i}: {self.reg.process(response)}'
        prompt += f'\n'
        return prompt

    def question_answer_prompt(self, question, clarifying_questions, context):
        prompt = 'I am trying to answer an arithmetic question. However, I have some clarification questions that I would like you to answer for me. Please provide detailed answers to my clarification questions.'
        prompt += f'\nQuestion: {question}?'
        prompt += f'\nClarification Questions: {self.reg.process(clarifying_questions)}'
        return prompt

    def refute_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to an arithmetic question. I would like you to analyze these responses and check for any errors, misunderstandings, or mathematical inaccuracies. Please clearly outline any errors or misunderstandings that you find. Additionally, you should provide details on how these errors and misunderstandings can be fixed. You should support your counter arguments with mathematical derivations.'
        prompt += f'\nQuestion: {question}?'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def helper_prompt(self, question, responses, context):
        prompt = ('I would like you to serve as an assistant and help me answer an arithmetic question. '
                  'I want you to help me by discussing the question and my answer. '
                  'You do not need to directly answer the question, but you should help guide me to produce a more accurate and correct answer.'
                  f'\nQuestion: {question}?'
                  f'\nMy Answer: {self.reg.process(responses[0])}'
        )
        return prompt
        




class GSMQuestion(Question):

    def __init__(self):
        super().__init__()

        self.Qtype = 'GSMQ'
        
        df0 = pd.read_parquet('GSM8K/train.parquet')
        df1 = pd.read_parquet('GSM8K/test.parquet')

        self.questions = list(df0['question']) + list(df1['question'])
        lang_answers   = list(df0['answer'])   + list(df1['answer'])

        self.answers = []

        pattern = r"#### (-?\d+\.?\d*)$"
        for answer in lang_answers:
            match = re.search(pattern, answer.replace(',', ''))
            if match:
                self.answers.append(float(match.group(1)))  # Convert the captured string to a float
            else:
                exit("invlaid GSM8K answer found in questions.QSMQuestion")



    def gen_question(self, num_questions=-1):
        if num_questions == -1:
            n = len(self.questions)
        else:
            n = min(num_questions, len(self.questions))
        for i in range(n):
            context = None
            yield self.questions[i], self.answers[i], context

    def basic_prompt(self, question, context=None, justification=True):
        prompt = (
            f'Please answer the following grade-school math question as accurately as possible. {question}'
            f'You should give a brief justification for your answer. You must provide your final answer as a single number by saying \"Final Answer:\".'
        )
        return prompt

    def summarize_prompt(self, responses, question):
        prompt = (f'Several people have provided several evaluations of the question \"{question}\". '
                  'I want you to provide a summarization of the given evaluations. '
                  'You should avoid commenting on the correctness and quality of each answer. '
                  'Your summary should preserve the person\’s the important components and additional details provided by the person.')
        for i, resp in enumerate(responses):
            prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        return prompt

    def debate_prompt(self, question, responses, summarized=False, context=None):
        prompt = 'Several people have provided responses to a grade-school math question. '
        if summarized:
            prompt += f'Here is a summary of their responses:\n{self.reg.process(responses)}'
        else:
            prompt += f'Blow are their responses:'
            for i, resp in enumerate(responses):
                prompt += f'\nPerson {i} said: {self.reg.process(resp)}'
        prompt += '\n\n'

        prompt += (
            f'You should take these responses into consideration when providing your own answer to the question. '
            f'You should give a brief justification for your answer. You must provide your final answer as a single number by saying \"Final Answer:\".'
            f'\nQuestion: {question}')

        return prompt

    def critic_prompt(self, question, response, context=None):
        prompt = (
            'I would like you to be the judge of an answer to a grade-school math question. You will be provided with a question and an answer to that question. '
            'Please evaluate the answer and identify any places in which the given answer is incorrect or inconsistent. '
            'If you identify any inconsistencies, please provide a list of specific details and discuss how the inconsistencies can be fixed.'
            f'\nQuestion: {question}')

        prompt += f'\n\nGiven Answer: {self.reg.process(response)}'

        return prompt

    def make_corrections_prompt(self, question, response, critiques, context=None):
        prompt = (
            'I would like you to make corrections to a given response. You will be given a question, a response to that question, and a list of possible issues with the given response. '
            'Please provide a corrected version of the given response based on the list of possible issues. You should try and make as few changes as possible to the original answer. '
            'It is very important that your corrected version still provides a final answer as a single number.'
            f'\nQuestion:  {question}')
        prompt += (
            f'\n\nGiven Response: {self.reg.process(response)}'
            f'\n\nPossible Issues: {self.reg.process(critiques)}')
        return prompt

    def devils_advocate_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a grade-school math question. I would like you to serve as the devil\'s advocate and provide counter arguments against these answers. You should support your counter arguments with accurate mathematical derivations.'
        prompt += f'\nQuestion: {question}'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def judge_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a grade-school math question. I would like you to serve as a judge of these answers. You should evaluate whether each answer is correct. You should support your evaluation with mathematical reasoning.'
        prompt += f'\nQuestion: {question}'

        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def detail_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a grade-school math question. I would like you to provide additional details that these answers may have missed. You should try to provide as many helpful detail as possible, along with accurate mathematical justification.'
        prompt += f'\nQuestion: {question}'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def context_prompt(self, question, responses, context):
        prompt = 'I would like you to help provide additional information about a grade-school math question. Without directly answering the question, you should provide additional details that you think would be helpful in answering the question. Please try to be as detailed as possible and as mathematically accurate as possible.'
        prompt += f'\nQuestion: {question}'
        return prompt

    def question_ask_prompt(self, question, responses, context):
        prompt = 'I would like you to help me improve a grade-school math question-answering task. You will be given the question and a list of responses to the question. You should ask clarifying questions about any part of the question or responses that you believe requires additional information or is not mathematically accurate.'
        prompt += f'\nQuestion: {question}'
        for i, response in enumerate(responses):
            prompt += f'\nResponse {i}: {self.reg.process(response)}'
        prompt += f'\n'
        return prompt

    def question_answer_prompt(self, question, clarifying_questions, context):
        prompt = 'I am trying to answer a grade-school math question. However, I have some clarification questions that I would like you to answer for me. Please provide detailed answers to my clarification questions.'
        prompt += f'\nQuestion: {question}'
        prompt += f'\nClarification Questions: {self.reg.process(clarifying_questions)}'
        return prompt

    def refute_prompt(self, question, responses, context):
        prompt = 'Several people have given answers to a grade-school math question. I would like you to analyze these responses and check for any errors, misunderstandings, or mathematical inaccuracies. Please clearly outline any errors or misunderstandings that you find. Additionally, you should provide details on how these errors and misunderstandings can be fixed. You should support your counter arguments with mathematical derivations.'
        prompt += f'\nQuestion: {question}'
        for i, response in enumerate(responses):
            prompt += f'\nPerson {i} answer: {self.reg.process(response)}'
        return prompt

    def helper_prompt(self, question, responses, context):
        prompt = ('I would like you to serve as an assistant and help me answer a grade-school math question. '
                  'I want you to help me by discussing the question and my answer. '
                  'You do not need to directly answer the question, but you should help guide me to produce a more accurate and correct answer.'
                  f'\nQuestion: {question}'
                  f'\nMy Answer: {self.reg.process(responses[0])}'
        )
        return prompt





