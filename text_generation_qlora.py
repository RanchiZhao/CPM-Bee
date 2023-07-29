from argparse import ArgumentParser
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from opendelta import LoraModel
import bmtrain as bmt
import torch
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--use-bminf", default=False, action="store_true", help="Whether to use BMInf")
    parser.add_argument("--memory-limit", type=int, default=5, help="GPU Memory limit, in GB")
    parser.add_argument("--delta", default=None, type=str, help="The path to lora.")
    parser.add_argument("--device", default="cuda:0", type=str, help="The target device.")
    args = parser.parse_args()
    return args

def load_quantize_state_dict(quantize_save):
    checkpoint = torch.load(quantize_save)
    state_dict = checkpoint["state_dict"]
    quant_state_dict = checkpoint["quant_state_dict"]
    for key, value in state_dict.items():
        if key in quant_state_dict:
            value.quant_state = quant_state_dict[key]

    return state_dict

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

def count_tokens(text, tokenizer):
    token_ids, _ = tokenizer.encode(text)
    return len(token_ids)


import os 

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    print(see_memory())

    data_list = [
        {"prompt": "以中华民族伟大复兴为主题写一篇1000字的文章。", "<ans>": ""}
   ]
    
    # bmt.init_distributed(seed=1234)
    # config = CPMBeeConfig.from_json_file("/root/zhaoyq/models/10b/cpm-bee-10b.json")
    # ckpt_path = "/root/gongbt/cpm-bee-hf/models/pytorch_model.bin"
    # tokenizer = CPMBeeTokenizer()
    # model = CPMBeeTorch(config=config)
    # model.load_state_dict(torch.load(ckpt_path), strict=False)

    bmt.init_distributed(seed=1234)
    config = CPMBeeConfig.from_json_file("/root/zhaoyq/models/10b/cpm-bee-10b.json")
    model = CPMBee(config)
    model.config = config
    ckpt_path = "/root/zhaoyq/models/10b/cpmbee_quantized.bin"
    tokenizer = CPMBeeTokenizer()
    state_dict = load_quantize_state_dict(ckpt_path)
    model.load_state_dict(state_dict)
    for name, param in model.named_parameters():
        if name in state_dict and hasattr(state_dict[name], 'quant_state'):
            param.quant_state = state_dict[name].quant_state
    
    if args.delta is not None:
        delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"], backend="bmt")
        model.load_state_dict(torch.load(args.delta), strict=False)

    if args.device == "cpu":
        model = model.float()
    else:
        if not torch.cuda.is_available():
            raise AssertionError("The CUDA is unavailable")
        if args.use_bminf:
            import bminf
            with torch.cuda.device(args.device):
                model = bminf.wrapper(model, quantization=False, memory_limit=args.memory_limit << 30)
        model.cuda(args.device)

    beam_search = CPMBeeBeamSearch(
        model=model,
        tokenizer=tokenizer,
    )

    print(see_memory())
    start_time = time.time()
    total_tokens = 0
    result_list = []

    for data in data_list:
        result = beam_search.generate([data], max_length=100, repetition_penalty=1.1)
        total_tokens += count_tokens(result[0]['<ans>'], tokenizer)  
        result_list.append(result)

    end_time = time.time()
    print(see_memory())
    elapsed_time = end_time - start_time
    print(f'生成了 {total_tokens} 个tokens, 耗时 {elapsed_time} 秒，推理速度为 {total_tokens / elapsed_time} tokens/秒')

    for res in result_list:
        print(res)

if __name__ == "__main__":
    main()
