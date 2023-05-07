#!/usr/bin/env bash
set -ex

DATA_DIR=${1}  # Required: the directory of the tokenized data file (spm: sentence piece)
PRETRAINED_MODEL_PATH=${2}  # Required: the filepath of the pretrained model checkpoint
OUTPUT_PATH=${3}
ARCH=${4}
UPDATE_FREQ=${5}
MAX_SENTENCES=${6}
LR=${7}
WARMUP_STEPS=${8}
SEED=${9}
MAX_EPOCH=${10}
CLIP_NORM=${11}
WEIGHT_DECAY=${12}
MAX_POSITIONS=${13}
INITIALIZATION_STRATEGY=${14}
SPLIT=${15}

if [ ! "${OUTPUT_PATH}" ]; then
    OUTPUT_PATH="./output/"
fi
if [ ! "${ARCH}" ]; then
    ARCH="electra_encoder_decoder_v6_base"
fi
if [ ! "${UPDATE_FREQ}" ]; then
    UPDATE_FREQ=8
fi
if [ ! "${MAX_SENTENCES}" ]; then
    MAX_SENTENCES=16
fi
if [ ! "${LR}" ]; then
    LR="3e-5"
fi
if [ ! "${WARMUP_STEPS}" ]; then
    WARMUP_STEPS=1000
fi
if [ ! "${SEED}" ]; then
    SEED=1
fi
if [ ! "${MAX_EPOCH}" ]; then
    MAX_EPOCH=4
fi
if [ ! "${CLIP_NORM}" ]; then
    CLIP_NORM=0.0
fi
if [ ! "${WEIGHT_DECAY}" ]; then
    WEIGHT_DECAY=0.1
fi
if [ ! "${MAX_POSITIONS}" ]; then
    MAX_POSITIONS=512
fi
if [ ! "${INITIALIZATION_STRATEGY}" ]; then
    INITIALIZATION_STRATEGY="discriminator"
fi
if [ ! "${SPLIT}" ]; then
    SPLIT="test"
fi

mkdir -p "${OUTPUT_PATH}"
echo "${OUTPUT_PATH}"

python train.py "${DATA_DIR}" \
  --task "xnli" \
  --finetune-from-model "${PRETRAINED_MODEL_PATH}" \
  --save-dir "${OUTPUT_PATH}" \
  --arch "${ARCH}" \
  --update-freq "${UPDATE_FREQ}" \
  --max-sentences "${MAX_SENTENCES}" \
  --lr "${LR}" \
  --warmup-updates "${WARMUP_STEPS}" \
  --seed "${SEED}" \
  --max-epoch "${MAX_EPOCH}" \
  --clip-norm "${CLIP_NORM}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --lr-scheduler "inverse_sqrt" \
  --optimizer "adam" --adam-betas "(0.9,0.98)" \
  --best-checkpoint-metric "accuracy" \
  --criterion "sentence_prediction" \
  --dropout 0.1 --attention-dropout 0.1 --pooler-dropout 0.1 \
  --max-positions "${MAX_POSITIONS}" \
  --rel-pos-buckets 32 --max-rel-pos 128 \
  --save-interval-updates 1000000 --keep-interval-updates 1 --save-interval 1 \
  --keep-best-checkpoints 5 --no-last-checkpoints \
  --maximize-best-checkpoint-metric \
  --init-token 0 --separator-token 2 --num-classes 3 --add-prev-output-tokens \
  --initialization-strategy "${INITIALIZATION_STRATEGY}" \
  --valid-subset "${SPLIT}" \
  --train-langs "en,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh" \
  --eval-langs "en,ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh" \
  --fp16 --ddp-backend=no_c10d \
  --log-interval 100 --log-format "simple" \
  --log-file "${OUTPUT_PATH}/train.log" \
  2>&1 | tee -a "${OUTPUT_PATH}/all_understanding.log"
