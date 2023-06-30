#! /bin/bash
export CUDA_VISIBLE_DEVICES=7

OPTS=""
OPTS+=" --use-delta"
OPTS+=" --model-config /root/zhaoyq/models/1b/cpm-bee-1b.json"
OPTS+=" --dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/train"
OPTS+=" --eval_dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/eval"
OPTS+=" --epoch 3"
OPTS+=" --batch-size 1"
OPTS+=" --train-iters 2"
OPTS+=" --save-name /root/zhaoyq/models/1b-finetuned"
OPTS+=" --max-length 2048"
OPTS+=" --save results/"
OPTS+=" --lr 0.0001"
OPTS+=" --inspect-iters 5"
OPTS+=" --warmup-iters 1"
OPTS+=" --eval-interval 1000"
OPTS+=" --early-stop-patience 5"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 32768"
OPTS+=" --start-step 0"
OPTS+=" --load /root/zhaoyq/models/1b/cpmbee_quantized.bin"
OPTS+=" --tensorboard /root/zhaoyq/tensorboard_log/qlora"

CMD="python /root/zhaoyq/CPM-Bee/src/finetune_cpm_bee_qlora.py ${OPTS}"

echo ${CMD}
$CMD

#OPTS+=" --load /root/zhaoyq/models/1b/cpmbee_quantized.bin"
#OPTS+=" --load /root/gongbt/cpm-bee-hf/models_1b/pytorch_model.bin"