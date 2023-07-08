from argparse import ArgumentParser
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from opendelta import LoraModel
import bmtrain as bmt
import torch

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

import os 

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args = parse_args()


    data_list = [
        {"input": "观山观水都能领略妙趣。", "options": {"<option_0>": "我来观水复观山", "<option_1>": "观水观山皆得妙", "<option_2>": "观书已若观山水", "<option_3>": "观水观山凭认取"}, "question": "这段话形容了哪句诗的意境？", "<ans>": ""},
    ]

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

    with open('/root/zhaoyq/model.txt', 'w') as f:
        for name, module in model.named_modules():
            f.write(f'Module name: {name}\n')
            f.write(f'Module type: {type(module).__name__}\n')
            
            for param_name, param in module.named_parameters():
                f.write(f'Parameter name: {param_name}\n')
                f.write(f'Parameter shape: {param.shape}\n')
                f.write(f'Parameter requires_grad: {param.requires_grad}\n')
                f.write(f'Parameter: {param[:10]}\n')
                try:
                    f.write(f'quant_state: {param.quant_state}\n')
                except:
                    raise ValueError
            f.write('\n') 
    
    if args.delta is not None:
        delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"], backend="bmt")
        model.load_state_dict(torch.load(args.delta), strict=False)

    if args.device == "cpu":
        #model = model.float()
        pass
    else:
        if not torch.cuda.is_available():
            raise AssertionError("The CUDA is unavailable")
        if args.use_bminf:
            import bminf
            with torch.cuda.device(args.device):
                model = bminf.wrapper(model, quantization=False, memory_limit=args.memory_limit << 30)
        model.cuda(args.device)

    # use beam search
    beam_search = CPMBeeBeamSearch(
        model=model,
        tokenizer=tokenizer,
    )
    inference_results = beam_search.generate(data_list, max_length=100, repetition_penalty=1.1)
    for res in inference_results:
        print(res)

if __name__ == "__main__":
    main()
