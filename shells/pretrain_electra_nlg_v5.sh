SPM_MODEL=/mnt/msranlp/shumma/data/16g/sentencepiece.bpe.model
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
APPEND_CMDS=${17}
GPUS=8
MAX_EPOCH=1000
CLIP_NORM=0
if [ ! $TEXT ]; then
    TEXT=/mnt/msranlp/shumma/data/16g/
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
ROOT=/mnt/output/PretrainedModels/electra-encoder-decoder-v5/
bsz=$((${MAX_SENTENCES}*${UPDATE_FREQ}*${NODES}*${GPUS}))
MODEL=$ROOT/lr${LR}-bsz${bsz}-ws${WARMUP_STEPS}-wd${WEIGHT_DECAY}-dw${DISCRIMINATOR_WEIGHT}_${DISCRIMINATOR_START_WARMUP_STEPS}_${DISCRIMINATOR_WARMUP_STEPS}-iw${INCONSISTENCY_WEIGHT}_${INCONSISTENCY_START_WARMUP_STEPS}_${INCONSISTENCY_WARMUP_STEPS}-g${GENERATOR_DECODER_LAYERS}d${DISCRIMINATOR_DECODER_LAYERS}${APPEND_CMDS}/
echo $MODEL
mkdir -p $MODEL
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $TEXT \
    --save-dir $MODEL --tensorboard-logdir $MODEL/logs --arch electra_encoder_decoder_v5_base --criterion electra_encoder_decoder_v5 \
    --discriminator-weight $DISCRIMINATOR_WEIGHT --discriminator-warmup-steps $DISCRIMINATOR_WARMUP_STEPS --inconsistency-weight $INCONSISTENCY_WEIGHT --inconsistency-warmup-steps $INCONSISTENCY_WARMUP_STEPS \
    --task pretraining --tokens-per-sample 512 --mask-prob 0.15 --span-length 3.0 --leave-unmasked-prob 0.0 --random-token-prob 0.0 --spm-model $SPM_MODEL \
    --share-all-embeddings --required-batch-size-multiple 8 --max-source-positions 1024 --max-target-positions 1024 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr $LR --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates $WARMUP_STEPS \
    --max-update $MAX_UPDATE --max-epoch $MAX_EPOCH --disable-validation --save-interval-updates 25000 --no-epoch-checkpoints \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --clip-norm $CLIP_NORM --weight-decay $WEIGHT_DECAY \
    --seed 1 --log-format simple --log-interval 100 --skip-invalid-size-inputs-valid-test --batch-read-ahead 10000 \
    --fp16 --fp16-init-scale 4 --ddp-backend=no_c10d \
    --rel-pos-buckets 32 --max-rel-pos 128 --rescale-init \
    --generator-decoder-layers ${GENERATOR_DECODER_LAYERS} --discriminator-decoder-layers ${DISCRIMINATOR_DECODER_LAYERS} \
    --discriminator-start-warmup-steps ${DISCRIMINATOR_START_WARMUP_STEPS} --inconsistency-start-warmup-steps ${INCONSISTENCY_START_WARMUP_STEPS} \
    --log-file $MODEL/train.log ${APPEND_CMDS}
# 2>&1 | tee -a $MODEL/train.log --fp16-scale-window 256 --scale-fc
