SPM_MODEL=/mnt/msranlp/shumma/data/160g/sentencepiece.bpe.model
TEXT=${1}
NODES=${2}
MAX_SENTENCES=${3}
UPDATE_FREQ=${4}
MAX_UPDATE=${5}
LR=${6}
WARMUP_STEPS=${7}
WEIGHT_DECAY=${8}
CLIP_NORM=${9}
SEED=${10}
MAX_SOURCE_POSITIONS=${11}
MAX_TARGET_POSITIONS=${12}
GPUS=8
MAX_EPOCH=100
if [ ! $TEXT ]; then
    TEXT=/mnt/msranlp/shumma/data/160g/
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
if [ ! $WARMUP_STEPS ]; then
    WEIGHT_DECAY=0.01
fi


MODEL=/mnt/output/PretrainedModels/t5/lr${LR}-bsz$((${MAX_SENTENCES}*${UPDATE_FREQ}*${NODES}*${GPUS}))-ws${WARMUP_STEPS}-wd${WEIGHT_DECAY}/
mkdir -p $MODEL
python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py $TEXT \
    --save-dir $MODEL --tensorboard-logdir $MODEL/logs --arch unilm_encoder_decoder_base --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --task pretraining --tokens-per-sample 512 --mask-prob 0.15 --span-length 3.0 --leave-unmasked-prob 0.0 --random-token-prob 0.0 --spm-model $SPM_MODEL \
    --share-all-embeddings --required-batch-size-multiple 8 --max-source-positions 1024 --max-target-positions 1024 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr $LR --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates $WARMUP_STEPS \
    --max-update $MAX_UPDATE --max-epoch $MAX_EPOCH --disable-validation --save-interval-updates 5000 --no-epoch-checkpoints \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --clip-norm $CLIP_NORM --weight-decay $WEIGHT_DECAY \
    --seed 1 --log-format simple --log-interval 100 --skip-invalid-size-inputs-valid-test --fp16 --batch-read-ahead 10000 \
    --ddp-backend=no_c10d --rel-pos-buckets 32 --max-rel-pos 128 2>&1 | tee -a $MODEL/train.log


#t5_transformer_base