import utility
import argparse
from llms import LLMHelper
from questions import BoolQuestion, MMLUQuestion, ArithmaticQuestion, GSMQuestion, BBHQuestion, SCIQQuestion, MedMCQAQuestion, ARCQuestion
from debaters import Debater
import torch

def single_model_on_dataset(model_name, params, role, is_support, question_set, num_questions, batch_size, resume=False, use_judge=False):
    """
    Test a model, with a given role (persona) on a dataset.
    All outputs and prompts are stored and saved in a dict.
    :param model_name: name of the model to use
    :param role: role (or persona) of the model
    :param is_support: boolean, if False, then the model must provide valid answers
    :param question_set: dataset of questions
    :param num_questions: number of total questions to answer via debate
    :param batch_size: number of debates to run in parallel
    :param resume: boolean, if True, then we load previously answered questions from memory
    :param use_judge: boolean, if True, calls GPT-4 as a judge (only used when the dataset does not have ground truth)
    :return: Nothing
    """

    # we test all models as debaters (this way we can test personas as well).
    # debater.batch_quest(r=0, question,...) is equivalent to debater.model.batch_query(basic_prompt(question),...)
    # i.e., setting round to 0 (r=0) makes debaters function as if they are normal LLMs.
    debater = Debater(model_name, params, question_set, llm_helper, role=role, is_support=is_support)

    # loads all questions, answer, and context (e.g., passages in BoolQ) from the dataset
    all_questions = [(question, answer, context) for question, answer, context in
                     question_set.gen_question(num_questions)]

    # logic to create batches
    num_questions = min(num_questions, len(all_questions))
    num_epochs = num_questions // batch_size
    if num_questions % batch_size != 0:
        num_epochs += 1

    d = {}  # holds performance results
    # loads previous questions
    save_name = f'{question_set.Qtype}/results/single/role={debater.role}_model={debater.model}'
    if resume  and utility.load_dict(save_name) is not None:
        loaded_d = utility.load_dict(save_name)
        for i in loaded_d:
            d[i] = loaded_d[i]
        print()
        print(f'Loaded {len(d)} questions from {save_name}')
        print()

    i = 0
    #####################################
    ######## debate block ###############
    #####################################
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

        batch = all_questions[batch_size * epoch: batch_size * (epoch + 1)]
        question_batch = [question for question, answer, context in batch]
        answer_batch   = [answer for question, answer, context in batch]
        context_batch  = [context for question, answer, context in batch]

        # r = 0 since we are not doing debate
        # this invokes basic_prompt in debater.batch_query
        r = 0

        # each has shape (batch_size, num_rounds)
        prompt_batch, resp_batch, pred_batch,  = debater.batch_query(r, question_batch, context_batch, question_set, previous_resp_batch=None,
                    max_trys=5, verbose=False)
        # saving outputs of debate
        for b in range(len(batch)):
            # i keeps track of the current question number (not the current batch index)
            d[i] = {'question': question_batch[b], 'answer': answer_batch[b], 'context': context_batch[b],
                    'prompt': prompt_batch[b], 'resp': resp_batch[b], 'pred': pred_batch[b]}

            d[i]['is_support'] = debater.is_support
            d[i]['roles'] = debater.role

        utility.save_dict(d, save_name)





if __name__ == '__main__':
    # Objects used to run the debate
    llm_helper = LLMHelper()
    reg = utility.RegExHelper()

    # creates a Question object for a given dataset
    # each dataset requires its own Question object to be written
    question_type_dict = {'BoolQ':   BoolQuestion,
                          'MMLUQ':   MMLUQuestion,
                          'MathQ':   ArithmaticQuestion,
                          'GSMQ':    GSMQuestion,
                          'BBH':     BBHQuestion,
                          'SCIQ':    SCIQQuestion,
                          'MedMCQA': MedMCQAQuestion,
                          'ARC':     ARCQuestion,
                          }

    parser = argparse.ArgumentParser()
    parser.add_argument("--Qtype",         type=str, help="Name of the dataset from which questions are constructed. Valid values: BoolQ, MMLUQ, MathQ, GSMQ, BBH, SCIQ, MedMCQA, ARC")
    parser.add_argument("--num_questions", type=int, help="Number of question to be asked (clipped to size of the dataset)")
    parser.add_argument("--batch_size", type=int, help="Number of debates to conduct in parallel")

    parser.add_argument("--role",       type=str,  help="Role of the single model to evaluate. In most cases this should be set to \"none\".")
    parser.add_argument("--model_name", type=str,  help="Name of the model to test.")
    parser.add_argument("--is_support", type=bool, help="Boolean, whether the model is serving a support role. Support models do not need to give valid answers.")

    parser.add_argument("--use_judge", type=bool, default=False, help="Boolean indicator, if true, then an LLM judge is used to evaluate correctness. Currently only supported for TruthQ")
    parser.add_argument("--resume",    type=bool, default=False, help="Boolean indicator, if true, then previously completed questions are loaded (and skipped). Useful instances where your job may be killed mid-run.")

    args = parser.parse_args()

    question_set = question_type_dict[args.Qtype]()

    assert not args.use_judge, "use_judge is only supported for judge-based datasets (not included in this release)."

    # default params for the models (we probably don't want to change this).
    params = {'temperature': 0.6,
              'max_tokens': 750,
              'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              }

    # verbose = 0: only very basic print-outs.
    # verbose = 1: prints out debate information without prompts and responses (e.g., number of invalid responses).
    # verbose = 2: prints out all information including prompts and responses.
    verbose = 0

    single_model_on_dataset(args.model_name, params, args.role, args.is_support, question_set, args.num_questions,
                            args.batch_size, resume=args.resume, use_judge=args.use_judge)
