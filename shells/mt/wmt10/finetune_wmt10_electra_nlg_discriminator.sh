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
MAX_SOURCE_POSITIONS=${13}
MAX_TARGET_POSITIONS=${14}
LANG_PAIRS=${15}
EXTRA_CMDS=${16}
DICT=/mnt/msranlp/shumma/data/deltalm/dict.txt
if [ ! $WARMUP_STEPS ]; then
    WARMUP_STEPS=1000
fi

if [ ! $MAX_EPOCH ]; then
    MAX_EPOCH=8
fi
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
LANGS="en,fr,cs,de,fi,lv,et,ro,hi,tr,gu"
#LANG_PAIRS="en-fr,fr-en,en-cs,cs-en,en-de,de-en,en-fi,fi-en,en-lv,lv-en,en-et,et-en,en-ro,ro-en,en-hi,hi-en,en-tr,tr-en,en-gu,gu-en"
GPUS=8
NODES=4
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $DATA_DIR \
    --task summarization_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --lang-pairs $LANG_PAIRS --langtoks '{"main":("tgt",None)}' --langs $LANGS \
    --finetune-from-model $PRETRAINED_MODEL_PATH --fixed-dictionary $DICT \
    --max-tokens $MAX_TOKENS \
    --tensorboard-logdir $OUTPUT_PATH/tb-log/ --log-file $OUTPUT_PATH/train_log.txt \
    --max-source-positions ${MAX_SOURCE_POSITIONS} --max-target-positions ${MAX_TARGET_POSITIONS} \
    --truncate-source \
    --share-all-embeddings \
    --arch $ARCH \
    --criterion electra_encoder_decoder_v6 \
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
    --save-interval-updates 100000 --start-save-epoch 1 \
    --fp16 --fp16-init-scale 4 --ddp-backend=no_c10d \
    --rel-pos-buckets 32 --max-rel-pos 128 ${EXTRA_CMDS}
#--no-epoch-checkpoints 
#--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_STEPS \
#--initialization-strategy "mlm_encoder2decoder" \
#--lr-scheduler inverse_sqrt --lr $LR --warmup-updates $WARMUP_STEPS \
#--lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_STEPS \




