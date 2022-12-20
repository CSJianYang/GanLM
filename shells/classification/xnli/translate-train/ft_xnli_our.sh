DATA_DIR=${1}
PRETRAINED_MODEL_PATH=${2}
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
LANG=${16}
EXTRA_CMDS=${17}
if [ ! $lr ]; then
    lr=5e-6
fi
if [ ! $update_freq ]; then
    update_freq=1
fi
if [ ! $bsz ]; then
    bsz=16
fi
GPUS=8
mkdir -p $OUTPUT_PATH
python train.py $DATA_DIR \
    --save-dir $OUTPUT_PATH  --finetune-from-model $PRETRAINED_MODEL_PATH \
    --max-sentences ${MAX_SENTENCES} --max-positions ${MAX_POSITIONS} --update-freq ${UPDATE_FREQ} --valid-subset ${SPLIT} \
    --train-langs "${LANG}" --eval-langs "${LANG}" \
    --task xnli --criterion sentence_prediction --arch ${ARCH} --log-interval 100 --log-format "simple" \
    --save-interval-updates 1000000 --keep-interval-updates 1 --save-interval 1 --keep-best-checkpoints 5 \
    --init-token 0 --separator-token 2 --num-classes 3 --add-prev-output-tokens --no-last-checkpoints \
    --dropout 0.1 --attention-dropout 0.1 --pooler-dropout 0.1 --weight-decay ${WEIGHT_DECAY} --clip-norm ${CLIP_NORM} \
    --optimizer adam --adam-betas "(0.9,0.98)" --lr-scheduler inverse_sqrt --lr ${LR} \
    --rel-pos-buckets 32 --max-rel-pos 128 --initialization-strategy ${INITIALIZATION_STRATEGY} \
    --warmup-updates ${WARMUP_STEPS} --max-epoch ${MAX_EPOCH} --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --ddp-backend=no_c10d --fp16 --log-file $OUTPUT_PATH/train.log 2>&1 | tee -a $OUTPUT_PATH/all.log