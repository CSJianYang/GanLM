#!/usr/bin/env bash
set -ex

DATA_DIR=${1}
PRETRAINED_MODEL_PATH=${2}
OUTPUT_PATH=${3}
ARCH=${4}
UPDATE_FREQ=${5}
MAX_TOKENS=${6}
LR=${7}
WARMUP_STEPS=${8}
SEED=${9}
MAX_EPOCH=${10}
CLIP_NORM=${11}
WEIGHT_DECAY=${12}

if [ ! "${UPDATE_FREQ}" ]; then
    UPDATE_FREQ=1
fi
if [ ! "${LR}" ]; then
    LR=5e-6
fi
if [ ! "${WARMUP_STEPS}" ]; then
    WARMUP_STEPS=1000
fi
if [ ! "${MAX_EPOCH}" ]; then
    MAX_EPOCH=10
fi

mkdir -p "${OUTPUT_PATH}"
echo "${OUTPUT_PATH}"

python train.py "${DATA_DIR}" \
  --task "seq2seq_generation" \
  --finetune-from-model "${PRETRAINED_MODEL_PATH}" \
  --save-dir "${OUTPUT_PATH}" \
  --arch "${ARCH}" \
  --update-freq "${UPDATE_FREQ}" \
  --max-tokens "${MAX_TOKENS}" \
  --lr "${LR}" \
  --warmup-updates "${WARMUP_STEPS}" \
  --seed "${SEED}" \
  --max-epoch "${MAX_EPOCH}" \
  --clip-norm "${CLIP_NORM}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lr-scheduler "inverse_sqrt" \
  --optimizer "adam" --adam-betas "(0.9,0.98)" --adam-eps 1e-08 \
  --best-checkpoint-metric "loss" \
  --criterion "electra_encoder_decoder_v6" --label-smoothing 0.1 \
  --dropout 0.1 --attention-dropout 0.1 \
  --max-source-positions 720 --max-target-positions 48 \
  --rel-pos-buckets 32 --max-rel-pos 128 \
  --max-update 1000000 --save-interval-updates 100000 --start-save-epoch 4 \
  --truncate-source --share-all-embeddings \
  --skip-invalid-size-inputs-valid-test --find-unused-parameters \
  --fp16 --fp16-init-scale 4 --ddp-backend=no_c10d \
  --no-progress-bar --log-interval 100 \
  --tensorboard-logdir "${OUTPUT_PATH}/tb-log/" \
  --log-file "${OUTPUT_PATH}/train_log.txt" \
  2>&1 | tee -a "${OUTPUT_PATH}/all_generation.log"
