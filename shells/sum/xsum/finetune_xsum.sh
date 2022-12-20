set -ex

BERT_MODEL_PATH=$1
DATA_DIR=$2
OUTPUT_PATH=$3
ARCH=$4
MAX_TOKENS=$5
LR=$6
WARMUP_STEPS=$7
SEED=$8


if [ ! $WARMUP_STEPS ]; then
  WARMUP_STEPS=500
fi

WEIGHT_DECAY="0.01"
BETAS="(0.9,0.999)"
CLIP="1.0"
TOTAL_NUM_UPDATES=50000
TOTAL_STEPS=50000
UPDATE_FREQ=1

mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH

python train.py $DATA_DIR \
    --finetune-from-model $BERT_MODEL_PATH \
    --max-tokens $MAX_TOKENS \
    --fp16 \
    --max-source-positions 512 --max-target-positions 64 \
    --task seq2seq_generation \
    --truncate-source \
    --share-all-embeddings \
    --arch $ARCH \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay $WEIGHT_DECAY --optimizer adam --adam-betas "$BETAS" --adam-eps 1e-08 \
    --clip-norm $CLIP \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_STEPS \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --best-checkpoint-metric loss \
    --max-update $TOTAL_STEPS --seed $SEED --save-dir ${OUTPUT_PATH} --no-progress-bar --log-interval 100 --no-epoch-checkpoints --no-last-checkpoints \
    --save-interval-updates 100000 \
    --ddp-backend=no_c10d \
    --rel-pos-buckets 32 --max-rel-pos 128
    | tee $OUTPUT_PATH/train_log.txt