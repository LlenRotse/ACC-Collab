import copy
import re
import pickle
import pandas as pd
import time
import argparse
import os
import json



# Save a dictionary into a file
def save_dict(dic, file_path):
    # Extract the directory part from the file_path
    directory = os.path.dirname(file_path)
    
    if any(type(key) == tuple for key in dic):
        dic = {str(key): dic[key] for key in dic}
        
    # Create the directory if it doesn't exist
    if directory:  # Check if the directory string is not empty
        os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w', encoding="utf-8") as file:
        # pickle.dump(dic, file)
        json.dump(dic, file, ensure_ascii=False, indent=4)


# Load a dictionary from a file
def load_dict(file_path, verbose=True, key_type='int'):
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            # loaded_dict = pickle.load(file)
            loaded_dict = json.load(file)
            if verbose:
                print(f"Dictionary loaded from {file_path}")
            if key_type=='int' and loaded_dict is not None:
                loaded_dict = {int(k): v for k, v in loaded_dict.items()}
            elif key_type=='int_tuple' and loaded_dict is not None:
                loaded_dict = {tuple([int(kk) for kk in k.replace(' ', '').replace('(', '').replace(')', '').split(',')]): v for k, v in loaded_dict.items()}
                
            return loaded_dict
    except Exception as e:
        if verbose:
            print(f"An error occurred while loading the dictionary: {e}")
        return None
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    



class RegExHelper:

    # class for regular expression checking
    def __init__(self):
        self.chars_to_remove = ',<>()!?\"\''

        self.percent_pattern = r'^\d*\.?\d+%$'
        self.float_pattern = r'^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$'
        self.fraction_pattern = r'^-?\d+/-?\d+$'

        self.yes_pattern = r'\byes\b'
        self.no_pattern  = r'\bno\b'
        self.total_query = 0
        self.invalid_query = 0



    def strip(self, s, chars):
        """
        removes all strings in chars from main_string, equivalent to
        :param s: a string
        :param chars: a list of chars
        :return: string
        """
        strip_s = copy.deepcopy(s)
        for c in chars:
            strip_s = strip_s.replace(c, '')
        return strip_s


    def is_float(self, s, must_be_0_1=False):
        """
        checks whether a given string is a float
        :param s: string
        :param must_be_0_1: boolean, determines if the float is required to be a probability
        :return: boolean
        """
        float_regex = re.compile(self.float_pattern)
        match = float_regex.match(s)
        if must_be_0_1 and match:
            return 0 <= float(s) <= 1
        else:
            return bool(match)

    def is_percent(self, s, must_be_0_100=False):
        """
        check whether a given string is a percentage
        :param s: string
        :param must_be_0_100: boolean, determines if the percentage must be between 0 and 100
        :return: boolean
        """
        match = re.match(self.percent_pattern, s)
        if match and must_be_0_100:
            return 0 <= float(s[:-1]) <= 100
        else:
            return bool(match)

    def is_fraction(self, s, must_be_0_1=False):
        """
        check whether a given string is a percentage
        :param s: string
        :param must_be_0_1: boolean, determines if the fraction must be a valid probability
        :return: boolean
        """
        match = re.match(self.fraction_pattern, s)
        if match:
            numerator, denominator = map(int, s.split('/'))
            if must_be_0_1 and denominator != 0:
                return 0 <= numerator / denominator <= 1
            else:
                return denominator != 0
        else:
            return False

    def remove_quoted_parts(self, s):
        """
        removes any parts of a string which are in quotes
        :param s: string
        :return: string
        """
        result = ""
        in_quotes = False

        for char in s:
            if char == "\"":
                in_quotes = not in_quotes
                continue
            if not in_quotes:
                result += char

        return result

    def format_response(self, response, strip_chars=None, remove_quotes=False, lower_case=False):
        """
        Formats the response of an LLM so that an answer can be extracted from the response (e.g., yes or no).
        :param response: the response of an LLM
        :param strip_chars: the characters to remove from the response (e.g., # or \n)
        :param remove_quotes: boolean, whether to remove quoted text within the response
        :param lower_case: boolean, whether to convert the response to all lower case
        :return: returns a formatted version of the response
        """
        resp = copy.deepcopy(response)
        if remove_quotes:
            resp = self.remove_quoted_parts(resp)

        resp = resp.replace('\n', ' ')
        resp = self.strip(resp, strip_chars)

        # removes all duplicate spaces
        resp = [rr for rr in resp.split(' ') if len(rr) > 0]
        resp = ' '.join(resp)

        if lower_case:
            resp = resp.lower()
        return resp




    def process(self, response):
        """
        specific version of format_response, mostly used for sentence embeddings (don't use this one)
        :param response:
        :return:
        """
        response = response.replace('\n', ' ')
        response = response.replace(':', ',')
        response = response.replace(';', ',')
        response = response.split(' ')
        response = [resp for resp in response if len(resp) > 0]
        response = ' '.join(response)
        return response

    def get_answer(self, response, Qtype, context=None, verbose=False):
        """
        generic get_answer function which will work with all datasets
        :param response: response of an LLM
        :param Qtype: question type (this determines which specific _get_answer_ function to call, e.g., BoolQ)
        :param context: additional information to be passed to the specific _get_answer_ function
        :param verbose: used for debugging
        :return: returns the extracted answer
        """

        if Qtype == 'BoolQ':
            val = self.get_yes_no_answer(response)
            if verbose and val is None:
                print('#' * 200)
                print('##### INVALID BOOLQ RESPONSE ######')
                print(response)
                print('#' * 200)
                with open('all_invalid.txt', 'a', encoding="utf-8") as fd:
                    fd.write(f'\n{response}\n##\n##\n')
                self.invalid_query += 1
            self.total_query += 1
            # print(f'TOTAL {self.total_query} and Invalid {self.invalid_query}')
                
            return val

        elif Qtype=='MMLUQ' or Qtype=='ARC' or Qtype=='SCIQ' or Qtype=='MedMCQA':
            val = self.get_multi_choice_answer(response, options=('a', 'b', 'c', 'd'))
            if verbose and val is None:
                print('#' * 200)
                print('##### INVALID multi-choice RESPONSE ######')
                print(response)
                print('#' * 200)
                self.invalid_query += 1
                with open('all_invalid.txt', 'a', encoding="utf-8") as fd:
                    fd.write(f'\n{response}\n##\n##\n')
            self.total_query += 1
            return val

        elif Qtype == 'BBH':
            if type(context) == int:
                val = self.get_yes_no_answer(copy.deepcopy(response))
                self.total_query += 1
                if val is None:
                    self.invalid_query += 1
                return val
            elif type(context) == str:
                val = self.get_multi_choice_answer(copy.deepcopy(response))
                self.total_query += 1
                if val is None:
                    self.invalid_query += 1
                return val
            
            val1 = self.get_multi_choice_answer(copy.deepcopy(response))
            val2 = self.get_yes_no_answer(copy.deepcopy(response))
            if verbose and ((val1 is None and val2 is None) or (val1 is not None and val2 is not None)):
                print('#' * 200)
                print('##### INVALID BBH RESPONSE ######')
                print(response)
                print('#' * 200)
                self.invalid_query += 1
                with open('all_invalid.txt', 'a', encoding="utf-8") as fd:
                    fd.write(f'\n{response}\n##\n##\n')
            self.total_query += 1
            if val1 is not None and val2 is     None:
                return val1
            if val1 is     None and val2 is not None:
                return val2


        elif Qtype == 'MathQ':
            val = self.get_math_answer(response, context)
            if verbose and val is None:
                print('#' * 200)
                print('##### INVALID Math RESPONSE ######')
                print(response)
                print('#' * 200)
                self.invalid_query += 1
            self.total_query += 1
            return val
        
        elif Qtype == 'GSMQ':
            val = self.get_math_answer(response, context)
            if verbose and val is None:
                print('#' * 200)
                print('##### INVALID GSM RESPONSE ######')
                print(response)
                print('#' * 200)
                self.invalid_query += 1
            self.total_query += 1
            return val

        

        elif Qtype == 'TruthQ':
            val = 0
            return val
        else:
            exit(f"Qtype={Qtype} is not implemented for RegExHelper.get_answer()")



    def get_batch_answer(self, response_list, Qtype, contexts=None, verbose=False):
        """
        gets answers for batched responses
        :param response_list: batch of LLM responses
        :param Qtype: question type of the dataset currently being used (e.g., BoolQ)
        :param contexts: batch of context
        :param verbose:  used in debugging
        :return:
        """
        answers = []

        if contexts is None:
            for response in response_list:
                answers.append(self.get_answer(response, Qtype, context=None, verbose=verbose))

        else:
            for response, context in zip(response_list, contexts):
                answers.append(self.get_answer(response, Qtype, context=context, verbose=verbose))

        return answers



    def get_yes_no_answer(self, response, return_resp=False, verbose=False):
        """
        Extracts a yes-no answer model the response of an LLM
        :param response: response of an LLM
        :return: 1 if yes, 0 if no, otherwise None
        """
        
        
        if 'answer: yes' in response.lower():
            return 1
        if 'answer: no' in response.lower():
            return 0
        
        resp = response.replace('\"Yes\"', 'Yes')
        resp = resp.replace('\"yes\"', 'yes')
        resp = resp.replace('\"No\"',  'No')
        resp = resp.replace('\"no\"',  'no')
        
        resp = resp.replace('\"Yes.\"', 'Yes')
        resp = resp.replace('\"yes.\"', 'yes')
        resp = resp.replace('\"No.\"',  'No')
        resp = resp.replace('\"no.\"',  'no')
        # format the response
        strip_chars = ['Mr.', 'Mrs.', 'Dr.', 'U.S.A.', 'U.S.',
                       ',', ':', '\'', '\"', '(', ')', 'a resounding', 'a firm', 'a definitive', 'Question', 'a clear',
                        # '-', '.', '?', '!'
                       '*']
        
        
        resp = self.format_response(resp, strip_chars, remove_quotes=True, lower_case=True)
        resp = resp.replace('question', 'question.')
        
        resp = [s for s in re.split(r'(?<=[.!?]) +', resp) if not s.strip().endswith('?')]
        resp = ' '.join(resp)
        
        resp = self.format_response(resp, ['.', '?', '!'], remove_quotes=False, lower_case=True)

    
        # list of response formats
        # manually collected by repeated trials with high temperature
        # fails to catch answers about 0.01% of the time (in this case the model is reprompted for a response)
        prefs = ['i would answer the question as', 'i would answer the question with',
                 'i will answer the question as', 'i will answer the question with',
                 'i would answer this question as', 'i would answer this question with',
                 'i will answer this question as', 'i will answer this question with',
                 'i believe the answer is', 'i believe that the answer is',
                 'i will answer this question as', 'i will answer this question with',
                 'my answer is', 
                 'my answer to the question is',
                 'my answer as',
                 'my answer as follows',
                 'i think the answer is',
                 'i would say',
                 'i would say that',
                 'i would answer',
                 'my answer to your question is',
                 'therefore the answer is',
                 'my final answer as',
                 'my final answer is',
                 'the answer to the question is',
                 'the correct answer is',
                 'my answer is',
                 'so the answer is',
                 'so my answer is',
                 'therefore the answer is',
                 'based on the context provided the answer is',
                 'based on the provided context the answer is',
                 'new answer',
                 'the final answer is',
                 'the corrected answer is',
                 'the answer to the question is',
                 'final answer',
                 'answer the question as follows',
                 'my answer is',
                 'my answer is also',
                 'the answer to the yes-no question is',
                 'the final answer to the question is',
                 'the final answer to the question on is',
                 'based on the passage the answer is',
                 'based on the passage provided the answer is',
                 'based on the given passage the answer is',
                 'based on the passage and the provided answers the answer is',
                 'based on the passage and the previous answers the answer is',
                 'based on the information provided in the passage and the responses given the answer is',
                 'based on the given information and the passage the answer is',
                 'based on the information provided in the passage the answer is',
                 'based on the given answers and the passage, the answer is',
                 'based on the information provided the answer is',
                 'the answer is',
                 'the answer to the question would be',
                 'my final answer to the question is',
                 'based on the given answers and the information in the passage the answer is',
                 'answer the question',
                 'answer to the question as',
                 'answer to the question is',
                 'answer to the question',
                 'answer the question with a',
                 'answer the question as',
                 'answer the question with',
                 'i would give a',
                 'my response to the question',
                 'a final answer of',
                 'i conclude that',
                 'my revised response',
                 'my answer to the question to',
                 'can be answered as',
                 'answer the following question as follows',
                 'answer the following question with a',
                 'answer the following question as',
                 'my answer to the question would be',
                 'i answer',
                 'my answer',
                 # 'answer',
                 
                 
                 
                 ]
        
        new_prefs = []
        for pref in prefs:
            if 'question' in pref and len(pref.split('question'))==2:
                prefix, suffix = pref.split('question')
                new_prefs.append(f'{prefix}yes-no question{suffix}')
        prefs += new_prefs
        
        for pref in prefs:
            if 'question' in pref and 'yes-no' not in pref and len(pref.split('question'))==2:
                prefix, suffix = pref.split('question')
                new_prefs.append(f'{prefix}following question{suffix}')
            elif 'yes-no question' in pref and len(pref.split('yes-no question'))==2:
                prefix, suffix = pref.split('yes-no question')
                new_prefs.append(f'{prefix}following yes-no question{suffix}')
        prefs += new_prefs


        # check for yes no answers within response, based on prefs
        is_yes, is_no = False, False
        for prefix in prefs:
            if prefix + ' no'  in resp:
                is_no = True
                if verbose:
                    print('IS NO: ', resp)
                    print('PREFIX: ', prefix)
                    print()
            if prefix + ' yes' in resp:
                is_yes = True
                if verbose:
                    print('IS Yes: ', resp)
                    print('PREFIX: ', prefix)
                    print()
        if is_yes  and not is_no:
            return 1
        elif is_no and not is_yes:
            return 0

        is_yes, is_no = False, False
        # if no patterns have caught the answer we check whether the first (or last) word is yes or no
        resp_split = resp.split(' ')
        if resp_split[0] == 'yes' or resp_split[-1] == 'yes':
            is_yes = True
        if resp_split[0] == 'no'  or resp_split[-1] == 'no':
            is_no = True
        # if resp_split[-1] == 'yes':
        #     is_yes = True
        # if resp_split[-1] == 'no':
        #     is_no = True

        if is_yes  and not is_no:
            return 1
        elif is_no and not is_yes:
            return 0
        
        if return_resp:
            return None, resp
        else:
            return None


    def get_multi_choice_answer(self, response, options=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p')):
        """
        Extracts a multiple choice answer from the response of an LLM
        :param response: response of an LLM
        :param options: possible options for the multiple choice answers (list of chars)
        :return: element of options which the LLM has selected (for consistency, this is always capitalized)
        """

        # format the response
        strip_chars = ['.', ',', ':', '\'', '\"', '!', '(', ')', '-', '?', '*']
        resp = self.format_response(response, strip_chars, remove_quotes=True, lower_case=True)

        prefs = [('the correct answer is', ''),
                 ('the correct option should be', ''),
                 ('final answer', ''),
                 ('the correct answer is indeed', ''),
                 ('final answer is', ''),
                 ('final answer', ''),
                 ('the correct answer is', ''),
                 ('correct answer should be', ''),
                 ('therefore the answer is', ''),
                 ('based on the information provided the answer is', ''),
                 ('the best answer is', ''),
                 ('therefore the answer is', ''),
                 ('', 'is my final answer'),
                 ('', 'is my answer'),
                 ('option', 'is the best'),
                 ('', 'is the best option'),
                 ('answer', 'is the best'),
                 ('', 'is the best answer'),
                 ('option', 'is the correct answer'),
                 ('my final answer is option', ''),
                 ('the answer to the question would be', ''),
                 ('option', 'is the most accurate'),
                 ('the final answer would be', ''),
                 ('the most likely answer is', ''),
                 ('will choose option', ''),
                 ('i have decided to choose', ''),
                 ('the most accurate answer is', ''),
                 ('', 'seems to be the most accurate'),
                 ('', 'is the most accurate'),
                 ('final answer', ''),
                 ]

        # loop through all patterns in prefs and extract the one of the answers in options
        found_answers = set()
        for prefix, suffix in prefs:
            for option in options:
                if   suffix == '':
                    term = f'{prefix} {option}'
                elif prefix == '':
                    term = f'{option} {suffix}'
                else:
                    term = f'{prefix} {option} {suffix}'

                if term in resp:
                    found_answers.add(option.capitalize())
        if len(found_answers)==1:
            return list(found_answers)[0]
            

        # if no pattern was found, we check if the first or last word in the response corresponds to an answer in options
        # resp_split = response.split(' ')
        # for option in options:
        #     if resp_split[0] == option or resp_split[-1] == option:
        #         return option.capitalize()

        return None


    def get_math_answer(self, response, expression=None):
        """
        Extracts answers for algebraic problems
        :param response: response of an LLM
        :param expression: algebraic expression being evaluated by the LLM (need to remove this expression in many cases)
        :return: a number
        """

        # formatting for math questions is slightly more tricky than _normal_ string-based questions
        resp = ""
        in_para = False

        # remove all text that is inside parenthesis
        for char in response:
            if char == "(":
                in_para = True
            if char == ")":
                in_para = False
                continue
            if not in_para:
                resp += char

        # models often repeat the expression (which can interfere with reg-ex)
        if expression is not None:
            resp = resp.replace(f'{expression} ', '')

        resp = resp.replace('\n', ' ')
        resp = resp.replace('+', ' ')
        resp = resp.replace('-', ' ')
        resp = resp.replace('*', ' ')
        resp = resp.replace('/', ' ')
        resp = resp.replace('=', ' ')
        resp = resp.replace('$', '')
        resp = resp.replace('£', '')
        resp = resp.replace('€', '')

        strip_chars = [',', ':', '\'', '\"', '!']
        resp = self.format_response(resp, strip_chars, remove_quotes=True, lower_case=True)


        patterns = [r'the correct answer to the expression is (\d+)',
                    r'the result of the expression is (\d+)',
                    r'the final answer is (\d+)',
                    r'(\d+) as the correct answer',
                    r'the correct answer to the expression is (\d+)',
                    r'(\d+) as the correct result of the expression',
                    r'my final answer is (\d+)',
                    r'final answer (\d+)',
                    ]

        for pattern in patterns:
            match = re.search(pattern, resp)
            if match:
                return int(match.group(1))

        if len(resp.split(' ')) > 0 and resp.split(' ')[-1].isdigit():
            return int(resp.split(' ')[-1])


        return None
















