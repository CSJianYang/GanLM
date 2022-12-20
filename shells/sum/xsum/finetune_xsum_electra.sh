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
EXTRA_CMDS=${13}


if [ ! $WARMUP_STEPS ]; then
  WARMUP_STEPS=1000
fi


mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH

python train.py $DATA_DIR \
    --finetune-from-model $PRETRAINED_MODEL_PATH \
    --max-tokens $MAX_TOKENS \
    --fp16 \
    --max-source-positions 720 --max-target-positions 48 \
    --task seq2seq_generation \
    --truncate-source \
    --share-all-embeddings \
    --arch $ARCH \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "(0.9,0.98)" --adam-eps 1e-08 \
    --clip-norm $CLIP_NORM \
    --lr-scheduler inverse_sqrt --lr $LR --warmup-updates $WARMUP_STEPS \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --best-checkpoint-metric loss \
    --max-epoch $MAX_EPOCH --seed $SEED --save-dir $OUTPUT_PATH --no-progress-bar --log-interval 100 \
    --save-interval-updates 100000 \
    --ddp-backend=no_c10d --rel-pos-buckets 32 --max-rel-pos 128 \
    | tee $OUTPUT_PATH/train_log.txt
#--initialization-strategy "mlm_encoder2decoder" \