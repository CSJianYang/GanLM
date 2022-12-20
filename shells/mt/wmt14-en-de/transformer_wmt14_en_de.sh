set -ex
#DATA_DIR=/mnt/msranlp/shumma/xsum/binary_data/en-unilm/
DATA_DIR=${1}
OUTPUT_PATH=${2}
ARCH=${3}
UPDATE_FREQ=${4}
MAX_TOKENS=${5}
LR=${6}
WARMUP_STEPS=${7}
SEED=${8}
MAX_EPOCH=${9}
CLIP_NORM=${10}
WEIGHT_DECAY=${11}
MAX_SOURCE_POSITIONS=${12}
MAX_TARGET_POSITIONS=${13}
src=${14}
tgt=${15}
EXTRA_CMDS=${16}
if [ ! $WARMUP_STEPS ]; then
    WARMUP_STEPS=1000
fi

if [ ! $MAX_EPOCH ]; then
    MAX_EPOCH=10
fi
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH

python train.py $DATA_DIR \
    --max-tokens $MAX_TOKENS \
    --tensorboard-logdir $OUTPUT_PATH/tb-log/ --log-file $OUTPUT_PATH/train_log.txt \
    --source-lang ${src} --target-lang ${tgt} \
    --max-source-positions ${MAX_SOURCE_POSITIONS} --max-target-positions ${MAX_TARGET_POSITIONS} \
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
    --max-update 1000000 --max-epoch $MAX_EPOCH --seed $SEED --save-dir ${OUTPUT_PATH} --no-progress-bar --log-interval 100 \
    --save-interval-updates 100000 --start-save-epoch 4 \
    --fp16 --fp16-init-scale 4 --ddp-backend=no_c10d \
    --rel-pos-buckets 32 --max-rel-pos 128 ${EXTRA_CMDS}
#--no-epoch-checkpoints 
#--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_STEPS \
#--initialization-strategy "mlm_encoder2decoder" \
#--lr-scheduler inverse_sqrt --lr $LR --warmup-updates $WARMUP_STEPS \
#--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_STEPS \