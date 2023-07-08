#! /bin/bash
export CUDA_VISIBLE_DEVICES=7

OPTS=""
OPTS+=" --use-bminf"
OPTS+=" --delta /root/zhaoyq/results/10b-finetuned-float32-delta-best.pt"

CMD="python /root/zhaoyq/CPM-Bee/src/text_generation_qlora.py ${OPTS}"

echo ${CMD}
$CMD