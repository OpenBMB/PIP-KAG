import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Any
import pdb 
from tqdm import tqdm
from datetime import datetime

# 获取当前日期
current_date = datetime.now()

# 格式化日期为 "25 Jan 2025" 格式
formatted_date = current_date.strftime("%d %b %Y")
print(f'The current is: {formatted_date}')


class SFTDataset(Dataset):
    def __init__(self, data_path , tokenizer, model_type, max_len, data_type):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_type = data_type

        if 'llama32' in model_type:
            format_name = 'llama3-2'
            self.system_format=''
            self.user_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: '+ str(formatted_date) + '\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistent<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>'
        elif 'llama3' in model_type:
            format_name = 'llama3'
            self.system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
            self.user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>\n'
        elif 'qwen2' in model_type:
            format_name = 'qwen2-5'
            self.system_format=''
            self.user_format='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n'
            self.assistant_format='{content}<|im_end|>\n'
        print(f'your chat template is :{format_name}')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_ids, target_mask = [], []
        
        # human = data['rag_input']
        # assistant = data['output']

        if self.data_type == 'w_context':
            human = self.user_format.format(content=data['rag_input'])
        elif self.data_type == 'wo_context':
            human = self.user_format.format(content=data['raw_input'])
        assistant = self.assistant_format.format(content=data['output'])

        # llama3没有system
        input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_len]
        target_mask = target_mask[:self.max_len]
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_mask,
        }


        return inputs
    

class SFTDataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']

            target_mask = x['labels']
            
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs
    

def compute_gradient(model, dataloader, device="cuda"):

    model.to(device)
    model.train()  # Set model to training mode

    # Use one batch from the dataloader
    inputs = next(iter(dataloader))
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    
    # Forward pass
    outputs = model(**inputs)
    loss = outputs.loss
    model.zero_grad()
    loss.backward()  # Backward pass to compute gradients

    gradients = {
        name: param.grad.detach().cpu()  # 转移到 CPU
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    return gradients


def compute_gradient_whole_batch(model, dataloader, device="cuda"):
    model.to(device)
    model.train()  # Set model to training mode

    gradient_accumulator = {
    name: torch.zeros_like(param, device=device)  # 初始化在 GPU 上
    for name, param in model.named_parameters()
    if param.requires_grad
}
    total_batches = 0

    for inputs in tqdm(dataloader, desc="Computing gradients", leave=True):
        total_batches += 1
        # 将输入数据转移到指定设备
        for key in inputs:
            inputs[key] = inputs[key].to(device)

        # 前向传播
        outputs = model(**inputs)
        loss = outputs.loss
        model.zero_grad()
        loss.backward()  # 反向传播计算梯度

        # 累积梯度
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_accumulator[name] += param.grad.detach()

    # 计算平均梯度
    gradients = {name: grad / total_batches for name, grad in gradient_accumulator.items()}
    gradients = {name: grad.cpu() for name, grad in gradients.items()}

    return gradients



def compute_context_sensitive_mask(model, w_context_dataloader, wo_context_dataloader, retain_ratio=0.8678, prune_type='layers'):

    # Compute gradients for w_context and wo_context
    # grad_w_context = compute_gradient(model, w_context_dataloader, device='cuda')
    # grad_wo_context = compute_gradient(model, wo_context_dataloader, device='cuda')
    # 
    grad_w_context = compute_gradient_whole_batch(model, w_context_dataloader, device='cuda')
    grad_wo_context = compute_gradient_whole_batch(model, wo_context_dataloader, device='cuda')
    
    print('开始计算sensitivity_diff')
    sensitivity_diff = {}
    if prune_type == 'model':
        # 全模型维度计算
        for name, param in model.named_parameters():
            if name in grad_w_context.keys():
                param_cpu = param.detach().cpu()
                grad_diff = grad_w_context[name] - grad_wo_context[name]
                sensitivity_diff[name] = torch.abs((grad_diff * param_cpu).float())
            else:
                raise KeyError(f"Parameter '{name}' is missing in grad_w_context.keys()")
    elif prune_type == 'layers':
        # 指定层数里计算
        start_layer, to_prune_layer_num = 23, 6
        for name, param in model.named_parameters():
            if 'layers' in name:
                layer_num = int(name.split('.layers.')[1].split('.')[0])
                if start_layer <= layer_num < start_layer + to_prune_layer_num:
                    if name in grad_w_context.keys():
                        param_cpu = param.detach().cpu()
                        grad_diff = grad_w_context[name] - grad_wo_context[name]
                        sensitivity_diff[name] = torch.abs((grad_diff * param_cpu).float())
                    else:
                        raise KeyError(f"Parameter '{name}' is missing in grad_w_context.keys()")
    
    all_scores = torch.cat([score.flatten() for score in sensitivity_diff.values()])
    # threshold = torch.quantile(all_scores, retain_ratio)
    print('开始计算阈值')
    num_params_to_keep = int(len(all_scores) * retain_ratio)  # 假设保留5%的连接
    print(f'keep param nums: {num_params_to_keep}')
    threshold, _ = torch.kthvalue(all_scores, num_params_to_keep)
    print(f'threshold: {threshold}')


    masks = {
        name: (score >= threshold).half()  # 
        for name, score in sensitivity_diff.items()
    }
    mask_sum = sum(mask.sum().item() for mask in masks.values())
    print(f'mask的数量: {mask_sum}')
    return masks



model_path = '/home/liuzhenghao/hpc/Models/llama/llama3/Llama3-8b-instruct'

# model_path = '/home/liuzhenghao/hpc/Models/llama/llama32/Llama-3.2-1B-inst'

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

data_path = '/home/liuzhenghao/hpc/project/lc_rag/datasets/my_kc/_Test/eval_mrqa_llm/cat_for_draw_acctivations_shuf1k_Mytrain.jsonl'
w_context_dataset = SFTDataset(data_path=data_path, tokenizer=tokenizer, model_type='llama3', max_len=1024, data_type='w_context')
wo_context_dataset = SFTDataset(data_path=data_path, tokenizer=tokenizer, model_type='llama3', max_len=1024, data_type='wo_context')

my_collator = SFTDataCollator(tokenizer=tokenizer, max_seq_length=1024)
w_context_dataloader = DataLoader(w_context_dataset, batch_size=1, collate_fn=my_collator)
wo_context_dataloader = DataLoader(wo_context_dataset, batch_size=1, collate_fn=my_collator)

retain_ratio = 0.3  # 保留 20% 的权重  0.8091
prune_type = 'layers'    # model 0.8678 layers 0.2
out_path = '/home/liuzhenghao/hpc/project/lc_rag/experiments/pruning_model/llama3_8b_instruct/snip/layers_23_6_03_mask_batch_data.pth'
masks = compute_context_sensitive_mask(model, w_context_dataloader, wo_context_dataloader, retain_ratio=retain_ratio, prune_type=prune_type)
torch.save(masks, out_path)

# class PrunedTrainer(Trainer):
#     def __init__(self, *args, masks=None, **kwargs):
#         """
#         Custom Trainer that integrates pruning masks.
#         """
#         super().__init__(*args, **kwargs)
#         self.masks = masks

#     def training_step(self, model, inputs):
#         """
#         Custom training step to reapply pruning masks after each optimizer step.
#         """
#         loss = super().training_step(model, inputs)

#         # Apply pruning masks to ensure sparsity
#         if self.masks:
#             with torch.no_grad():
#                 for name, param in model.named_parameters():
#                     if name in self.masks:
#                         param.data *= self.masks[name]

#         return loss


# if __name__ == "__main__":
#     # Load model and tokenizer
#     model_name = "Meta/Llama-3b"
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     # Load dataset
#     dataset = load_dataset("json", data_files="/path/to/dataset.json", split="train")
#     def preprocess_function(examples):
#         return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)
#     dataset = dataset.map(preprocess_function, batched=True)

#     # Prepare dataloader for SNIP pruning
#     dataloader = DataLoader(dataset, batch_size=4)

#     # Apply SNIP pruning
#     retain_ratio = 0.2
#     masks = snip_prune(model, dataloader, retain_ratio=retain_ratio)

#     # Define training arguments
#     training_args = TrainingArguments(
#         output_dir="./results",
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         gradient_accumulation_steps=2,
#         evaluation_strategy="steps",
#         save_steps=500,
#         save_total_limit=2,
#         fp16=True,
#         logging_dir="./logs",
#         logging_steps=100,
#         report_to="tensorboard"
#     )

#     # Initialize PrunedTrainer
#     trainer = PrunedTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         tokenizer=tokenizer,
#         masks=masks
#     )

#     # Train the model
#     trainer.train()
