#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

GPUS_PER_NODE=4

NNODES=1
MASTER_ADDR="localhost"
MASTER_PORT=12345

OPTS=""
OPTS+=" --model-config /root/zhaoyq/models/10b/cpm-bee-10b.json"
OPTS+=" --load  /root/gongbt/cpm-bee-hf/models/pytorch_model.bin"
OPTS+=" --teacher-config /root/zhaoyq/models/10b/cpm-bee-10b.json"
OPTS+=" --load-teacher  /root/gongbt/cpm-bee-hf/models/pytorch_model.bin"
OPTS+=" --dataset /root/zhaoyq/CPM-Bee/tutorials/basic_task_finetune/bin_data/train"
OPTS+=" --cook-config /root/zhaoyq/BMCook/examples/cpm_live/configs/cpm-bee.json"
OPTS+=" --save-name /root/zhaoyq/models/cook/cooked_model.bin"
OPTS+=" --epoch 3"
OPTS+=" --tensorboard /root/zhaoyq/tensorboard_log/bmcook"
OPTS+=" --batch-size 2"
OPTS+=" --max-length 2048"
OPTS+=" --lr 0.0001"
OPTS+=" --warmup-iters 1"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 0.01"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 32768"
OPTS+=" --start-step 0"

CMD="torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} /root/zhaoyq/CPM-Bee/src/cpm_bee_test.py ${OPTS}"

echo ${CMD}
$CMD


