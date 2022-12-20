SPM_MODEL=/mnt/msranlp/shumma/data/deltalm/spm.model
DICT_FILE=/mnt/msranlp/shumma/data/deltalm/dict.txt
TEXT=${1}
NODES=${2}
MAX_SENTENCES=${3}
UPDATE_FREQ=${4}
MAX_UPDATE=${5}
LR=${6}
WARMUP_STEPS=${7}
WEIGHT_DECAY=${8}
DISCRIMINATOR_WEIGHT=${9}
DISCRIMINATOR_WARMUP_STEPS=${10}
INCONSISTENCY_WEIGHT=${11}
INCONSISTENCY_WARMUP_STEPS=${12}
GENERATOR_DECODER_LAYERS=${13}
DISCRIMINATOR_DECODER_LAYERS=${14}
DISCRIMINATOR_START_WARMUP_STEPS=${15}
INCONSISTENCY_START_WARMUP_STEPS=${16}
DISCRIMINATOR_SAMPLING_TEMPERATURE=${17}
SEED=${18}
MAX_SOURCE_POSITIONS=${19}
MAX_TARGET_POSITIONS=${20}
APPEND_CMDS=${21}
GPUS=8
MAX_EPOCH=1000
CLIP_NORM=1.0
if [ ! $TEXT ]; then
    TEXT=/mnt/msranlp/shumma/data/deltalm/
fi
if [ ! $NODES ]; then
    NODES=4
fi
if [ ! $MAX_SENTENCES ]; then
    MAX_SENTENCES=32
fi
if [ ! $UPDATE_FREQ ]; then
    UPDATE_FREQ=1
fi
if [ ! $LR ]; then
    LR=1e-4
fi
if [ ! $WARMUP_STEPS ]; then
    WARMUP_STEPS=4000
fi
if [ ! $WEIGHT_DECAY ]; then
    WEIGHT_DECAY=0.01
fi
if [ ! $DISCRIMINATOR_WEIGHT ]; then
    DISCRIMINATOR_WEIGHT=25
fi
if [ ! $DISCRIMINATOR_WARMUP_STEPS ]; then
    DISCRIMINATOR_WARMUP_STEPS=-1
fi
if [ ! $GENERATOR_DECODER_LAYERS ]; then
    GENERATOR_DECODER_LAYERS=12
fi
ROOT=/mnt/output/PretrainedModels/multilingual/electra-encoder-decoder-v6/
bsz=$((${MAX_SENTENCES}*${UPDATE_FREQ}*${NODES}*${GPUS}))
MODEL=$ROOT/lr${LR}-bsz${bsz}-ws${WARMUP_STEPS}-wd${WEIGHT_DECAY}-dw${DISCRIMINATOR_WEIGHT}_${DISCRIMINATOR_START_WARMUP_STEPS}_${DISCRIMINATOR_WARMUP_STEPS}-iw${INCONSISTENCY_WEIGHT}_${INCONSISTENCY_START_WARMUP_STEPS}_${INCONSISTENCY_WARMUP_STEPS}-g${GENERATOR_DECODER_LAYERS}d${DISCRIMINATOR_DECODER_LAYERS}${APPEND_CMDS}/
#MODEL=/mnt/output/PretrainedModels/electra-encoder-decoder-v6/poly/lr6e-4-bsz8192-ws10000-wd0.01-dw10_1000_-1-iw0.5_1000_-1-g12d1--share-generator-discriminator/
echo $MODEL
mkdir -p $MODEL
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $TEXT \
    --save-dir $MODEL --tensorboard-logdir $MODEL/logs --arch electra_encoder_decoder_v6_base --criterion electra_encoder_decoder_v6 \
    --discriminator-weight $DISCRIMINATOR_WEIGHT --discriminator-warmup-steps $DISCRIMINATOR_WARMUP_STEPS --inconsistency-weight $INCONSISTENCY_WEIGHT --inconsistency-warmup-steps $INCONSISTENCY_WARMUP_STEPS \
    --task pretraining --tokens-per-sample 512 --mask-prob 0.15 --span-length 3.0 --leave-unmasked-prob 0.0 --random-token-prob 0.0 --spm-model $SPM_MODEL --dict-file $DICT_FILE \
    --share-all-embeddings --required-batch-size-multiple 8 --max-source-positions ${MAX_SOURCE_POSITIONS} --max-target-positions ${MAX_TARGET_POSITIONS} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler polynomial_decay --total-num-update $MAX_UPDATE --lr $LR --warmup-updates $WARMUP_STEPS \
    --max-update $MAX_UPDATE --max-epoch $MAX_EPOCH --disable-validation --save-interval-updates 5000 --no-epoch-checkpoints \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --clip-norm $CLIP_NORM --weight-decay $WEIGHT_DECAY \
    --seed ${SEED} --log-format simple --log-interval 100 --skip-invalid-size-inputs-valid-test --batch-read-ahead 10000 \
    --fp16 --fp16-init-scale 4 --ddp-backend=no_c10d \
    --rel-pos-buckets 32 --max-rel-pos 128 --rescale-init \
    --generator-decoder-layers ${GENERATOR_DECODER_LAYERS} --discriminator-decoder-layers ${DISCRIMINATOR_DECODER_LAYERS} --discriminator-sampling-temperature ${DISCRIMINATOR_SAMPLING_TEMPERATURE} \
    --discriminator-start-warmup-steps ${DISCRIMINATOR_START_WARMUP_STEPS} --inconsistency-start-warmup-steps ${INCONSISTENCY_START_WARMUP_STEPS} \
    --log-file $MODEL/train.log ${APPEND_CMDS}
# 2>&1 | tee -a $MODEL/train.log --fp16-scale-window 256 --scale-fc polynomial_decay inverse_sqrt
 