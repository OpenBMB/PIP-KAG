from typing import List
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig, LlamaForInputContrastivew_act_inhibit
# from transformers import AutoModelForCausalLM, AutoConfig

from peft import get_peft_model, LoraConfig
import torch

def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")
    elif model_type in ["llava", "llava_next", "llava_next_video", "mllama", "paligemma", "video_llava"]:
        forbidden_modules.add("multi_modal_projector")
    elif model_type == "qwen2_vl":
        forbidden_modules.add("merger")

    if freeze_vision_tower:
        if model_type == "mllama":
            forbidden_modules.add("vision_model")
        elif model_type == "qwen2_vl":
            forbidden_modules.add("visual")
        else:
            forbidden_modules.add("vision_tower")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    print("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def load_model_and_tokenizer(model_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path if not model_args.tokenizer_name else model_args.tokenizer_name, trust_remote_code=True, use_fast=False) # llama不支持use fast
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_type = training_args.model_type
    if model_type in ['LlamaForCausalLM_w_act_inhibit', 'LlamaForInputContrastivew_act_inhibit', 'Qwen2ForInputContrastivew_act_inhibit']:
        if model_type == 'LlamaForCausalLM_w_act_inhibit':
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.architectures= ['LlamaForCausalLM_w_act_inhibit']
            model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch.bfloat16,  # 使用 BF16
                    config=config,
                    trust_remote_code=True,
                    inhibit_strength= training_args.inhibit_strength,
                    inhibit_layer_list= training_args.inhibit_layer_list,
            )            
        elif model_type == 'LlamaForInputContrastivew_act_inhibit':
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.architectures= ['LlamaForInputContrastivew_act_inhibit'] 
            model = LlamaForInputContrastivew_act_inhibit.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch.bfloat16,  # 使用 BF16
                    config=config,
                    trust_remote_code=True,
                    inhibit_strength= training_args.inhibit_strength,
                    inhibit_layer_list= training_args.inhibit_layer_list,
            )
        elif model_type == 'Qwen2ForInputContrastivew_act_inhibit':
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.architectures= ['Qwen2ForInputContrastive_w_act_inhibit']
            model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch.bfloat16,  # 使用 BF16
                    config=config,
                    trust_remote_code=True,
                    inhibit_strength= training_args.inhibit_strength,
                    inhibit_layer_list= training_args.inhibit_layer_list,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,  # 使用 BF16
                trust_remote_code=True        )
        
    if training_args.use_lora == False:
        peft_config = None
    elif training_args.use_lora == True:
        # 找到所有需要插入adapter的全连接层
        # target_modules = find_all_linear_names(model, training_args.train_mode)
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            # task_type="CAUSAL_LM",
            inference_mode=False,
            bias="none",
        )
        # import pdb; pdb.set_trace()
        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()  # 这行对 LoRA 很重要
        print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()
    else:
        raise ValueError(f'train_mode {training_args.train_mode} not supported')
    
    return model, tokenizer

