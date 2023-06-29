# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Dict, List, Union
import torch
import bmtrain as bmt
import os
from opendelta import LoraModel
from cpm_live.arguments import get_args
from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from cpm_live.utils import allgather_objects
from cpm_live.training_tasks.bee import FinetuneDataset

### add here
import bitsandbytes as bnb
import logging
import importlib_metadata
from packaging import version
from copy import deepcopy
from transformers import BitsAndBytesConfig
import torch.nn as nn

import os
import psutil
# from accelerate import init_empty_weights
keep_in_fp32_modules = None
quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,#
            load_in_8bit=False,#
            llm_int8_threshold=6.0,#
            llm_int8_has_fp16_weight=False,#
            bnb_4bit_compute_dtype=torch.float32,#
            bnb_4bit_use_double_quant=True,#
            bnb_4bit_quant_type='nf4',  # {'fp4', 'nf4'}
            llm_int8_skip_modules = None
        )

from cpm_live.layers.linear import Linear

def replace_with_bnb_linear(model, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        # print(type(module)) 
        if isinstance(module, Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            # if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
            if True:
                # with init_empty_weights():
                if quantization_config.quantization_method() == "llm_int8":
                    model._modules[name] = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        None,
                        has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                        threshold=quantization_config.llm_int8_threshold,
                    )
                else:
                    if (
                        quantization_config.llm_int8_skip_modules is not None
                        and name in quantization_config.llm_int8_skip_modules
                    ):
                        pass
                    else:
                        import gc
                        print("1:",see_memory())
                        
                        new_layer = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            # module.bias is not None,
                            None,
                            quantization_config.bnb_4bit_compute_dtype,
                            compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                            quant_type=quantization_config.bnb_4bit_quant_type,
                        )
                        # new_layer = nn.Linear(module.in_features,module.out_features).to(dtype=torch.float16)
                        new_layer = new_layer.cuda()
                        
                        model._modules[name] = new_layer
                        for param_name, param in module.named_parameters():
                            print(f'Parameter shape: {param.shape}\n')
                        print("2:",see_memory())
                        # exit(0)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        # Remove the last key for recursion
        if len(list(module.children())) > 0:
            replace_with_bnb_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
            )
    return model

def get_keys_to_not_convert(model):
    # Create a copy of the model and tie the weights, then
    # check if it contains tied weights
    tied_model = deepcopy(model)  # this has 0 cost since it is done inside `init_empty_weights` context manager`
    tied_model.tie_weights()

    tied_params = find_tied_parameters(tied_model)
    # For compatibility with Accelerate < 0.18
    if isinstance(tied_params, dict):
        tied_keys = list(tied_params.values())
    else:
        tied_keys = sum([x[1:] for x in tied_params], [])
    has_tied_params = len(tied_keys) > 0

    # Check if it is a base model
    is_base_model = not hasattr(model, model.base_model_prefix)

    # Ignore this for base models (BertModel, GPT2Model, etc.)
    if (not has_tied_params) and is_base_model:
        return []

    # otherwise they have an attached head
    list_modules = list(model.named_parameters())
    list_last_module = [list_modules[-1][0]]

    # add last module together with tied weights
    intersection = set(list_last_module) - set(tied_keys)
    list_untouched = tied_keys + list(intersection)

    # remove ".weight" from the keys
    names_to_remove = [".weight", ".bias"]
    filtered_module_names = []
    for name in list_untouched:
        for name_to_remove in names_to_remove:
            if name_to_remove in name:
                name = name.replace(name_to_remove, "")
        filtered_module_names.append(name)

    return filtered_module_names

def apply_quantization(model, quantization_config, device_map=None):
    # llm_int8_skip_modules = quantization_config.llm_int8_skip_modules
    # load_in_8bit_fp32_cpu_offload = quantization_config.llm_int8_enable_fp32_cpu_offload

    # logger = logging.getLogger(__name__)
    # logger.info("Detected 8-bit loading: activating 8-bit loading for this model")

    # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
    if quantization_config.llm_int8_skip_modules is None:
        # modules_to_not_convert = get_keys_to_not_convert(model)
        modules_to_not_convert = None
        pass
    else:
        modules_to_not_convert = llm_int8_skip_modules

    if not isinstance(modules_to_not_convert, list):
        modules_to_not_convert = [modules_to_not_convert]

    # modules_to_not_convert.extend(keep_in_fp32_modules)

    # Extend the modules to not convert to keys that are supposed to be offloaded to `cpu` or `disk`
    if isinstance(device_map, dict) and len(device_map.keys()) > 1:
        keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

        if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
            raise ValueError("If you want to offload some keys to `cpu` or `disk`, you need to set "
                             "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                             "converted to 8-bit but kept in 32-bit.")

        modules_to_not_convert.extend(keys_on_cpu)

    supports_4bit = version.parse(importlib_metadata.version("bitsandbytes")) >= version.parse("0.39.0")

    if quantization_config.load_in_4bit and not supports_4bit:
        raise ValueError("You have a version of `bitsandbytes` that is not compatible with 4bit inference and training."
                         "Make sure you have the latest version of `bitsandbytes` installed.")

    model = replace_with_bnb_linear(model, modules_to_not_convert=modules_to_not_convert,
                                    quantization_config=quantization_config)

    # # training in 8-bit is only available in 0.37.0+
    # model._is_kbit_training_enabled = version.parse(importlib_metadata.version("bitsandbytes")) >= version.parse("0.37.0")

    return model

def get_tokenizer(args):
    tokenizer = CPMBeeTokenizer()
    return tokenizer

def see_cpu_memory():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'resident': round(memory_info.rss / (1024 * 1024 * 1024), 2),  # GB
        'virtual': round(memory_info.vms / (1024 * 1024 * 1024), 2),  # GB
    }

from contextlib import contextmanager,ExitStack
_init_weights = True
@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    TODO(Patrick): Delete safety argument `_enable=True` at next major version. .
    """
    global _init_weights
    old_init_weights = _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = old_init_weights

from typing import ContextManager
class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)

def get_model(args):
    config = CPMBeeConfig.from_json_file(args.model_config)
    print("before_model_init: ",see_memory())
    print("before_init:  ", see_cpu_memory())
    # from accelerate import init_empty_weights
    # init_contexts = [no_init_weights(_enable=True)]
    # init_contexts.append(init_empty_weights())
    # with ContextManagers(init_contexts):
    
    model = CPMBee(config)

    print("after_init: ", see_cpu_memory())
    print("after_model_init: ",see_memory())

    model = apply_quantization(model,quantization_config=quantization_config)
    print("after_quan: ",see_memory())

    # for name, module in model.named_modules():
    #     for param_name, param in module.named_parameters():
    #         print(f'Parameter shape: {param.shape} parameter dtype: {param.dtype}\n')

    print_model_dtype(model)
    # model.config = config
    # if args.load is not None:
    #     bmt.load(model, args.load)
    # else:
    #     bmt.init_parameters(model)

    print("after_load: ",see_memory())

    total_param_size = 0
    for param in model.parameters():
        total_param_size += param.element_size() * param.nelement()
    total_param_size = total_param_size / (1024 ** 3)
    print("total_param_size: ", total_param_size, "GB")
    
    with open('/root/zhaoyq/model.txt', 'w') as f:
        for name, module in model.named_modules():
            f.write(f'Module name: {name}\n')
            f.write(f'Module type: {type(module).__name__}\n')
            
            for param_name, param in module.named_parameters():
                f.write(f'Parameter name: {param_name}\n')
                f.write(f'Parameter shape: {param.shape}\n')
                f.write(f'Parameter requires_grad: {param.requires_grad}\n')
            
            f.write('\n') 
    exit(0)
    #bmt.save(model, "/root/zhaoyq/models/1b/quantized.pt")
    
    # cast all non INT8 parameters to fp32
    # for param in model.parameters():
    #     if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
    #         param.data = param.data.to(torch.float32)
            
    # insert LoRA
    if args.use_delta:
        delta_model = LoraModel(
            backbone_model=model, modified_modules=["project_q", "project_v"], backend="bmt"
        )
        delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
        delta_model.log()
    print("after_lora: ",see_memory())

    print_model_dtype(model)
    # print_model_dtype(model)
        
    # for name, module in model.named_modules():
    #     print(name)
    #     try:
    #         print('dtype: ', module.weight.dtype) #float16
    #     except:
    #         print('pass')
        # if isinstance(module, LoraLayer):
        #     if args.bf16:
        #         module = module.to(torch.bfloat16)
        # if 'lora' in name:
        #     # if args.bf16:
        #     module = module.to(torch.bfloat16)
        # if 'norm' in name:
        #     # print(name)
        #     print('Before conversion:', module.weight.dtype) #float16
        #     module = module.to(torch.float32)
        #     print('After conversion:', module.weight.dtype) #float32
        # if 'lm_head' in name or 'embed_tokens' in name:#input_embedding
        #     if hasattr(module, 'weight'):
        #         # if args.bf16 and module.weight.dtype == torch.float32:
        #         module = module.to(torch.bfloat16)#torch.float32
    total_param_size = 0
    for param in model.parameters():
        total_param_size += param.element_size() * param.nelement()
    total_param_size = total_param_size / (1024 ** 3)
    print("total_param_size: ", total_param_size, "GB")
    bmt.save(model, "/root/zhaoyq/models/1b/quantized.pt")
    return model

def print_model_dtype(model):
    for name, module in model.named_modules():
        print("Model name: ", name)
        
        if hasattr(module, 'weight') and module.weight is not None:
            print('Weight dtype: ', module.weight.dtype)
            
        if hasattr(module, 'bias') and module.bias is not None:
            print('Bias dtype: ', module.bias.dtype)

        print("-"*20)

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(
        model.parameters(), weight_decay=args.weight_decay
    )
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_scheduler = bmt.lr_scheduler.Noam(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler

def setup_model_and_optimizer(args):
    model = get_model(args)
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    optim_manager = bmt.optim.OptimManager(
        loss_scale=args.loss_scale,
        loss_scale_factor=2,
        loss_scale_steps=512,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)
    return tokenizer, model, optimizer, lr_scheduler, optim_manager

def initialize():
    args = get_args(finetune=True)
    bmt.init_distributed(seed=args.seed)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    return args

def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res

def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage

def evaluation(model, args, tokenizer, loss_func):
    bmt.print_rank("evaluation begins...")
    eval_dataloader = FinetuneDataset(
        args.eval_dataset,
        1,
        args.max_length,
        tokenizer,
        max_depth=8,
        task_name=args.task_name,
        drop_last=args.drop_last,
    )
    eval_losses = []
    last_data = None

    with torch.no_grad():
        for iteration, data in enumerate(eval_dataloader):
            iteration = iteration + 1
            skip_this_batch = False
            if data is None:
                if last_data is None:
                    raise RuntimeError(
                        "Dataset is too small, please use a smaller batch size or sequence length!"
                    )
                data = last_data
                skip_this_batch = True
            else:
                last_data = data

            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_ids_sub = torch.from_numpy(data["inputs_sub"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            input_context = torch.from_numpy(data["context"]).cuda().bool()
            input_sample_ids = torch.from_numpy(data["sample_ids"]).cuda().to(torch.int32)
            input_num_segments = torch.from_numpy(data["num_segments"]).cuda().to(torch.int32)
            input_segment_ids = torch.from_numpy(data["segment_ids"]).cuda().to(torch.int32)
            input_segment_rel_offset = (
                torch.from_numpy(data["segment_rel_offset"]).cuda().to(torch.int32)
            )
            input_segment_rel = torch.from_numpy(data["segment_rel"]).cuda().to(torch.int32)
            input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            ext_table_ids = torch.from_numpy(data["ext_ids"]).cuda().to(torch.int32)
            ext_table_sub = torch.from_numpy(data["ext_sub"]).cuda().to(torch.int32)
            # ===========
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # ===========
            logits, _ = model(
                input_ids,
                input_ids_sub,
                input_length,
                input_context,
                input_sample_ids,
                input_num_segments,
                input_segment_ids,
                input_segment_rel_offset,
                input_segment_rel,
                input_span,
                ext_table_ids,
                ext_table_sub,
            )

            loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
            if skip_this_batch:
                loss = loss * 0
            eval_losses.append(bmt.sum_loss(loss))

        overall_loss = torch.stack(eval_losses).mean().item()
    return overall_loss

def finetune(
    args,
    tokenizer: CPMBeeTokenizer,
    model: CPMBee,
    optimizer: bmt.optim.AdamOffloadOptimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
    optim_manager: bmt.optim.OptimManager,
):

    average_time = bmt.utils.AverageRecorder()
    if model.config.dtype == torch.half:
        loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    else:
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if args.tensorboard is not None and bmt.rank() == 0:
        from torch.utils.tensorboard import SummaryWriter
        import distutils.version  # noqa: F401

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    best_eval_loss, eval_loss_increase = 1e9, 0
    global_token_pass = 0.0
    global_steps = 0
    global_world_size = bmt.world_size()
    dataloader = FinetuneDataset(
        args.dataset,
        args.batch_size,
        args.max_length,
        tokenizer,
        max_depth=8,
        task_name=args.task_name,
        drop_last=args.drop_last,
    )

    print("before epoch: ",see_memory())

    def print_layer_type_and_dtype(module, input, output):
        print(type(module), input[0].dtype)

    # hook
    for module in model.modules():
        module.register_forward_hook(print_layer_type_and_dtype)

    for epoch in range(args.epoch):
        epoch = epoch + 1
        last_data = None
        for iteration, data in enumerate(dataloader):
            iteration = iteration + 1
            global_steps = global_steps + 1
            skip_this_batch = False
            if data is None:
                if last_data is None:
                    raise RuntimeError(
                        "Dataset is too small, please use a smaller batch size or sequence length!"
                    )
                data = last_data  # use last data
                skip_this_batch = True
            else:
                last_data = data

            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_ids_sub = torch.from_numpy(data["inputs_sub"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            input_context = torch.from_numpy(data["context"]).cuda().bool()
            input_sample_ids = torch.from_numpy(data["sample_ids"]).cuda().to(torch.int32)
            input_num_segments = torch.from_numpy(data["num_segments"]).cuda().to(torch.int32)
            input_segment_ids = torch.from_numpy(data["segment_ids"]).cuda().to(torch.int32)
            input_segment_rel_offset = (
                torch.from_numpy(data["segment_rel_offset"]).cuda().to(torch.int32)
            )
            input_segment_rel = torch.from_numpy(data["segment_rel"]).cuda().to(torch.int32)
            input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            ext_table_ids = torch.from_numpy(data["ext_ids"]).cuda().to(torch.int32)
            ext_table_sub = torch.from_numpy(data["ext_sub"]).cuda().to(torch.int32)
            task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            task_names = data["task_names"]
            # ===========
            optim_manager.zero_grad()
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # ===========
            logits, _ = model(
                input_ids,
                input_ids_sub,
                input_length,
                input_context,
                input_sample_ids,
                input_num_segments,
                input_segment_ids,
                input_segment_rel_offset,
                input_segment_rel,
                input_span,
                ext_table_ids,
                ext_table_sub,
            )
            loss = loss_func(logits.view(-1, logits.size(-1)), targets.long().view(-1))
            if skip_this_batch:
                loss = loss * 0

            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            # ===========
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)
            # ===========
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_norm=1.0)
            optim_manager.step()
            mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)
            # ==========
            iteration_time = tim_usage["optim"] - tim_usage["init"]
            average_time.record(iteration_time)

            with torch.no_grad():
                task_num = len(task_names)
                targets_tmp = targets.expand(task_num, -1, -1)
                task = torch.arange(task_num, dtype=torch.int32, device="cuda")[:, None, None]
                targets_tmp = torch.where(
                    task_ids == task,
                    targets_tmp,
                    torch.scalar_tensor(-100, dtype=torch.int32, device="cuda"),
                )

                task_loss_map: Dict[str, float] = {}
                if not skip_this_batch:
                    for i in range(task_num):
                        task_loss = loss_func(
                            logits.view(-1, logits.size(-1)), targets_tmp[i, :].long().view(-1)
                        )
                        task_loss_map[task_names[i]] = task_loss.item()
                gatherd_task_loss_map: List[Dict[str, float]] = allgather_objects(task_loss_map)

                global_task_loss_map: Dict[str, Union[List[float], float]] = {}
                for local_task_loss_map in gatherd_task_loss_map:
                    for task_name, task_loss in local_task_loss_map.items():
                        if task_name not in global_task_loss_map:
                            global_task_loss_map[task_name] = []
                        global_task_loss_map[task_name].append(task_loss)

                task_loss_map = {}
                for task_name in sorted(list(global_task_loss_map.keys())):
                    avg_loss = sum(global_task_loss_map[task_name]) / len(
                        global_task_loss_map[task_name]
                    )
                    task_loss_map[task_name] = avg_loss

            local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
            local_total_rate = bmt.sum_loss(local_total_rate).item()
            global_token_pass += (
                global_world_size * local_total_rate * args.max_length * args.batch_size
            )
            avg_time = average_time.value

            train_info = {
                "time": tim_usage["init"],
                "epoch": epoch,
                "iteration": iteration,
                "loss": task_loss_map[args.task_name],
                "lr": lr_scheduler.current_lr,
                "lr_scale": int(optim_manager.loss_scale),
                "time_usage": tim_usage,
                "mem_usage": mem_usage,
                "avg_time": avg_time,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / avg_time,
                "grad_norm": grad_norm.item(),
                "mask_max": ((targets >= 0).sum(-1).float().mean() / args.max_length).item(),
                "num_gpus": global_world_size,
                "task_loss": task_loss_map,
            }

            bmt.print_rank(
                (
                    "| Epoch: {:3d} | Iter: {:6d} | loss: {:.4f} "
                    + "| lr: {:.4e}, scale: {:10.4f} | time: {:.4f} |"
                    + " token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.10f}"
                ).format(
                    epoch,
                    iteration,
                    task_loss_map[args.task_name],
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    avg_time,
                    input_length.float().mean() / args.max_length,
                    (targets >= 0).sum(-1).float().mean() / args.max_length,
                    grad_norm,
                )
            )
            bmt.print_rank(
                "| "
                + " | ".join(
                    [
                        "{} loss: {:.4f}".format(task_name, loss)
                        for task_name, loss in task_loss_map.items()
                    ]
                )
            )
            if iteration % args.inspect_iters == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                bmt.print_rank(bmt.inspect.format_summary(model_inspect))
                train_info["model_inspect"] = model_inspect
                print(train_info["mem_usage"])

            # write log here
            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", task_loss_map[args.task_name], global_steps)
                for task_name, loss in task_loss_map.items():
                    writer.add_scalar("Loss/train/{}".format(task_name), loss, global_steps)

            # evaluation
            if global_steps % args.eval_interval == 0:
                eval_loss = evaluation(model, args, tokenizer, loss_func)
                if args.tensorboard is not None and bmt.rank() == 0:
                    writer.add_scalar("Loss/eval", eval_loss, global_steps)
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    eval_loss_increase = 0
                    if args.save is not None:
                        if not args.use_delta:
                            bmt.save(model, os.path.join(args.save, args.save_name + "-best.pt"))
                        else:
                            state_dict = model.state_dict()
                            if bmt.rank() == 0:
                                torch.save(state_dict, os.path.join(args.save, args.save_name + "-delta-best.pt"))
                else:
                    eval_loss_increase += 1
                bmt.print_rank(
                    "| Eval loss: {:.4f} | Increase: {:2d}".format(eval_loss, eval_loss_increase)
                )
                if eval_loss_increase == args.early_stop_patience:
                    bmt.print_rank(
                        "Eval loss has increased {:d} times, the finetune loop early stopped."
                        .format(eval_loss_increase)
                    )
                    return
    # end of finetune

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler, optim_manager = setup_model_and_optimizer(args)
    print("before finetune:",see_memory())
    finetune(args, tokenizer, model, optimizer, lr_scheduler, optim_manager)

if __name__ == "__main__":
    main()
