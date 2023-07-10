## CPM-Bee单卡QLoRA微调

### 使用CPM-Bee进行基础任务量化微调

本教程在**使用CPM-Bee进行基础任务微调**的基础上，对模型引入量化操作，在保证模型训练效果的前提下降低显存消耗。经测试，此方法支持RTX3090 24GB单卡上对CPM-Bee-10B模型的全精度int4量化微调。

首先，你需要对模型参数文件进行量化调整。

进入工作路径：

```bash
$ cd src
```

量化调整参数文件：

```bash
$ python quantize_state_dict.py --input-path your_cpmbee_model.bin --output-path your_cpmbee_quantize_model.bin
```

其次，你需要根据需求手动设置config文件。

下面的例子代表采用全精度+int4量化；网络前向计算时转换为torch.float32；采用双重量化；量化类型为nf4。

```json
    "half" : false, 
    "load_in_4bit" : true,
    "compute_dtype" : "torch.float32",
    "compress_statistics" : true,
    "quant_type" : "nf4"
```

最后，其余部分就可以参考基础微调教程来完成。

注意在你的微调脚本中记得讲`--load`替换为

`your_cpmbee_quantize_model.bin`

另外，由于dtype的限制，你需要注释掉inspect部分的代码，

```python
if iteration % args.inspect_iters == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                bmt.print_rank(bmt.inspect.format_summary(model_inspect))
                train_info["model_inspect"] = model_inspect
                print(train_info["mem_usage"])
```

否则会报错：

`not available for std and var only support floating point and complex dtypes`

