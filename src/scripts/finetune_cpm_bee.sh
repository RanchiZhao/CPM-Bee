#! /bin/bash
export CUDA_VISIBLE_DEVICES=7

OPTS=""
OPTS+=" --use-delta"
OPTS+=" --model-config /root/zhaoyq/models/10b/cpm-bee-10b.json"
OPTS+=" --dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/train"
OPTS+=" --eval_dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/eval"
OPTS+=" --epoch 3"
OPTS+=" --batch-size 4"
OPTS+=" --save-name 10b-finetuned-float16"
OPTS+=" --max-length 2048"
OPTS+=" --save /root/zhaoyq/results/"
OPTS+=" --lr 0.0001"
OPTS+=" --inspect-iters 5"
OPTS+=" --warmup-iters 1"
OPTS+=" --eval-interval 50"
OPTS+=" --early-stop-patience 10"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 32768"
OPTS+=" --start-step 0"
OPTS+=" --load /root/zhaoyq/models/10b/cpmbee_quantized.bin"
OPTS+=" --tensorboard /root/zhaoyq/tensorboard_log/qlora_float16"

CMD="python /root/zhaoyq/CPM-Bee/src/finetune_cpm_bee_qlora.py ${OPTS}"

echo ${CMD}
$CMD