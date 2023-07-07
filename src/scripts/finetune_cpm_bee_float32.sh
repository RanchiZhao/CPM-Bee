#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

OPTS=""
OPTS+=" --use-delta"
OPTS+=" --model-config /root/zhaoyq/models/10b/cpm-bee-10b.json"
OPTS+=" --dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/train"
OPTS+=" --eval_dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/eval"
OPTS+=" --epoch 1"
OPTS+=" --batch-size 2"
OPTS+=" --train-iters 3000"
OPTS+=" --save-name 10b-finetuned-float32"
OPTS+=" --max-length 2048"
OPTS+=" --save /root/zhaoyq/results/"
OPTS+=" --lr 0.0001"
OPTS+=" --inspect-iters 5"
OPTS+=" --warmup-iters 1"
OPTS+=" --eval-interval 100"
OPTS+=" --early-stop-patience 5"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 32768"
OPTS+=" --start-step 0"
OPTS+=" --load /root/zhaoyq/models/10b/cpmbee_quantized.bin"
OPTS+=" --tensorboard /root/zhaoyq/tensorboard_log/qlora_float32"

CMD="python /root/zhaoyq/CPM-Bee/src/finetune_cpm_bee_qlora.py ${OPTS}"

echo ${CMD}
$CMD