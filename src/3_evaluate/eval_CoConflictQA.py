import re
import ast
import string
import json
import jsonlines
import re
import argparse
from rouge import Rouge
from tqdm import tqdm
import os
import random
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
# from transformers import Llama_pruning_ffnForCausalLM, Llama_pruning_attnForInputContrastive, Llama_pruning_attnForCausalLM, Qwen2_pruning_ffnForCausalLM, Llama_pruning_ffn_ByList_ForCausalLM
import logging
# from vllm import LLM, SamplingParams
import math

# IS_INSTRUCTION_PROMPT = False
print('='*20 + f'卡的数量:{torch.cuda.device_count()}' + '='*20)
def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'true', 't', 'yes', 'y', '1'}:
        return True
    elif value.lower() in {'false', 'f', 'no', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def call_llama(model, tokenizer, input_res, max_new_tokens):
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # with torch.no_grad():
    with torch.inference_mode():
        sequences = model.generate(input_res, max_new_tokens = max_new_tokens, pad_token_id=tokenizer.eos_token_id)[0, input_res.shape[-1]:]
        decoded = tokenizer.decode(sequences, skip_special_tokens=True)

        output_str = decoded.strip()

        return output_str
          

def run_vllm_infer(model_path, gpu_utils,  max_new_tokens, in_prompt_list):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=gpu_utils, max_model_len=4096, tensor_parallel_size=torch.cuda.device_count(), max_num_seqs=1)
    print(max_new_tokens)


    max_batch_size = 1024
    all_prompts = in_prompt_list
    
    sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=max_new_tokens)
    all_preds = []
    for i in tqdm(range(0, len(all_prompts), max_batch_size), total=math.ceil(len(all_prompts) / max_batch_size), leave=False, desc=f"vllm infering"):
        j = min(i+max_batch_size, len(all_prompts))
        cur_prompts = all_prompts[i: j]
        chat_prompt_list = []
        for tmp in cur_prompts:
            user_input = [ {"role": "user", "content": tmp}]
            user_input = tokenizer.apply_chat_template(user_input, add_generation_prompt=True, tokenize=False)
            chat_prompt_list.append(user_input)

        outputs_list = llm.generate(chat_prompt_list, sampling_params)

        for o_idx, outputs in enumerate(outputs_list):
            
            outputs = outputs.outputs[0]
            texts = outputs.text
            all_preds.append(texts)
    return all_preds

                
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))



def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def _acc_score(prediction, ground_truth):
    # if ground_truth in prediction or ground_truth.lower() in prediction or ground_truth.capitalize() in prediction:
    if normalize_answer(ground_truth) in normalize_answer(prediction):
        return 1.0
    else:
        return 0.0

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(normalize_answer(prediction), normalize_answer(ground_truth), avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_score(preds, golds):
    acc_score, rouge_score, f1_score, em_score = 0, 0, 0, 0
    for pred, gold in zip(preds, golds):
        
        if isinstance(gold, list): 
            for g in gold:
                _acc, _rouge, _f1, _em = 0,0,0,0
                _acc = max(_acc_score(pred, g), _acc)
                _rouge = max(_rougel_score(pred, g), _rouge)
                _f1 = max(_f1_score(pred, g), _f1)
                _em = max(_exact_match_score(pred, g), _em)
        else:
            _acc = _acc_score(pred, gold)
            _rouge = _rougel_score(pred, gold)
            _f1 = _f1_score(pred, gold)
            _em = _exact_match_score(pred, gold)
    
        acc_score += _acc
        rouge_score += _rouge
        f1_score += _f1
        em_score += _em
    acc_score = acc_score * 100 / (len(preds) + 1e-5)
    rouge_score = rouge_score * 100 / (len(preds) + 1e-5)
    f1_score = f1_score * 100 / (len(preds) + 1e-5)
    em_score = em_score * 100 / (len(preds) + 1e-5)
    return acc_score, rouge_score, f1_score, em_score


def get_score_with_paramatric(paramatric_answers, preds, golds):
    pm_acc_score, acc_score, rouge_score, f1_score, em_score = 0, 0, 0, 0, 0
    for pm_answer, pred, gold in zip(paramatric_answers, preds, golds):
        
        if isinstance(gold, list): 
            for g in gold:
                _acc, _rouge, _f1, _em = 0,0,0,0
                _acc = max(_acc_score(pred, g), _acc)
                _rouge = max(_rougel_score(pred, g), _rouge)
                _f1 = max(_f1_score(pred, g), _f1)
                _em = max(_exact_match_score(pred, g), _em)
        else:
            _acc = _acc_score(pred, gold)
            _rouge = _rougel_score(pred, gold)
            _f1 = _f1_score(pred, gold)
            _em = _exact_match_score(pred, gold)
    
        _pm_acc = _acc_score(pred, pm_answer)
        if _pm_acc == 1:
            _acc = 0

        pm_acc_score += _pm_acc
        acc_score += _acc
        rouge_score += _rouge
        f1_score += _f1
        em_score += _em

    pm_acc_score = pm_acc_score * 100 / (len(preds) + 1e-5)
    acc_score = acc_score * 100 / (len(preds) + 1e-5)
    rouge_score = rouge_score * 100 / (len(preds) + 1e-5)
    f1_score = f1_score * 100 / (len(preds) + 1e-5)
    em_score = em_score * 100 / (len(preds) + 1e-5)
    print('='*40)
    print('pm_acc_score, acc_score, rouge_score, f1_score, em_score')
    return pm_acc_score, acc_score, rouge_score, f1_score, em_score


def qa_to_prompt(query, context):
    prompt = '{}\nQ: {}\nA: '.format(context, query)
    return prompt

def qa_to_prompt_baseline(query, context, schema, tokenizer, IS_INSTRUCTION_PROMPT):
    def get_prompt(query, context, schema, answer=''):
        if schema == 'base_wo_context':
            prompt = 'Q: {}\nA: {}'.format(query, answer)
        # elif 
        elif schema == 'base':
            prompt = '{}\nQ: {}\nA: {}'.format(context, query, answer)
        elif schema == 'opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA: {}'.format(context, query[:-1], answer)
        elif schema == 'instr+opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'attr':
            prompt = '{}\nQ:{} based on the given tex?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr':
            prompt = '{}\nQ: {}\nA: {}'.format(context, query, answer)
        return prompt
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    if IS_INSTRUCTION_PROMPT:
        prompt = [{'role': 'system', 'content': prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return prompt

    
def eval(parametric_answers, pred_answers, gold_answers, step):
    # acc, rouge, f1, em = get_score(pred_answers, gold_answers)
    pm_acc, acc, rouge, f1, em = get_score_with_paramatric(parametric_answers, pred_answers, gold_answers)

    mr = round((pm_acc / (pm_acc+acc+1e-5)) * 100, 2)
    acc = round(acc, 2)
    rouge = round(rouge, 2)
    f1 = round(f1, 2)
    em = round(em, 2)

    logging.info('Step: {}: pc {}, po {}, mr {}, em {}.'.format(step, acc, pm_acc, mr, em))
    return acc, pm_acc, mr, em
    
def create_log_path(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('') 
        logging.info(f"Log file {log_path} created.")
    else:
        logging.info(f"Log file {log_path} already exists.")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="./Models/Qwen2-7B-Instruct", type=str)
    parser.add_argument("--data_path", default="./ConFiQA/ConFiQA-QA.json", type=str)
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--output_path", default='./result/Qwen2-7B-Instruct.json', type=str)
    parser.add_argument("--log_path", default='./log_ConFiQA/Qwen2-7B-Instruct.log' , type=str)
    parser.add_argument('--use_chat_template', type=str2bool, nargs='?', default=False, help='Enable or disable the flag')
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--act_inhibit_ratio", default=1.0, type=float)
    parser.add_argument("--act_inhibit_layer_list",
                            type=int,            # 或 float
                            nargs='+',           # 表示可接收多个值
                            default=[None],         # 默认值是 list
                            help="List of layers to inhibit"
                        )
    args = parser.parse_args()
    print(f'################ IS_INSTRUCTION_PROMPT is :{args.use_chat_template} ################')
    model_name = args.model_name

    base_path = os.path.dirname(args.output_path)
    os.makedirs(base_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_path),  # 写入日志文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info("Evaluate Context-Faithfulness for the Model: %s" % model_name)
    logging.info(f"schema {args.schema}")
    

    with jsonlines.open(args.data_path, 'r') as reader:
        data = list(reader)

    logging.info('Loaded {} instances.'.format(len(data)))
    
    create_log_path(args.log_path)
    
    gold_answers, pred_answers, all_pm_answers = [], [], []

    # tokenizer_path = '/home/liuzhenghao/hpc/Models/llama/llama3/Llama3-8b-instruct'
    tokenizer_path = model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"####### max_new_tokens :{args.max_new_tokens} #######")
    
    logging.info("####### using generate for infer #######")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if args.act_inhibit_layer_list != 0:
        if 'llama' in config.architectures[0].lower():
            config.architectures= ['LlamaForCausalLM_w_act_inhibit']
            print('using LlamaForCausalLM_w_act_inhibit')
        elif 'qwen2' in config.architectures[0].lower():
            config.architectures= ['Qwen2ForCausalLM_w_act_inhibit']
            print('using Qwen2ForCausalLM_w_act_inhibit')
        else:
            raise ValueError('unsupported model type, pleas contact the author')
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map="auto", 
                                                 low_cpu_mem_usage = True, 
                                                 torch_dtype=torch.bfloat16, 
                                                 config=config,
                                                 trust_remote_code=True, 
                                                 inhibit_strength=args.act_inhibit_ratio, 
                                                 inhibit_layer_list=args.act_inhibit_layer_list)
    model.eval()
    step = 0
    input_ids_list = []
    for d in data:
        query = d['question']
        context = d['context']
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema, tokenizer=tokenizer, IS_INSTRUCTION_PROMPT=args.use_chat_template)

        input_ids_list.append(tokenizer(prompt, return_tensors='pt').input_ids)
    input_ids_list = [ids.to(model.device) for ids in input_ids_list]
    for data_idx, d in tqdm(enumerate(data), total=len(data)):
        step += 1
        answer = d['answers']
        parametric_answer = d['parametric_answer']
        all_pm_answers.append(parametric_answer)
        gold_answers.append(answer)
        
            # pred = call_llama(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        pred = call_llama(model, tokenizer, input_ids_list[data_idx], max_new_tokens=args.max_new_tokens)
            
        pred_answers.append(pred)
        d['pred'] = pred
        
        if step % 500 == 0:
            eval(all_pm_answers, pred_answers, gold_answers, step)
    logging.info('final evaluation....')
    
    final_acc, final_rouge, final_f1, final_em = eval(all_pm_answers, pred_answers, gold_answers, step)
    final_acc, final_rouge, final_f1, final_em = round(final_acc, 2), round(final_rouge, 2), round(final_f1, 2), round(final_em, 2)
    logging.info(f'{final_acc}\t{final_rouge}\t{final_f1}\t{final_em}')
    
    base_path = os.path.dirname(args.output_path)
    os.makedirs(base_path, exist_ok=True)
    with jsonlines.open(args.output_path, mode='w') as writer:
        for d in data:
            writer.write(d)
        

if __name__ == '__main__':
    main()
