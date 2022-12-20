set -ex
NODES=${1}
DATA_DIR=${2}
PRETRAINED_MODEL_PATH=${3}
OUTPUT_PATH=${4}
ARCH=${5}
UPDATE_FREQ=${6}
MAX_TOKENS=${7}
LR=${8}
WARMUP_STEPS=${9}
SEED=${10}
MAX_EPOCH=${11}
CLIP_NORM=${12}
WEIGHT_DECAY=${13}
MAX_SOURCE_POSITIONS=${14}
MAX_TARGET_POSITIONS=${15}
STRATEGY=${16}
EXTRA_CMDS=${17}
DICT=/mnt/msranlp/shumma/data/deltalm/dict.txt
if [ ! $WARMUP_STEPS ]; then
    WARMUP_STEPS=1000
fi

if [ ! $MAX_EPOCH ]; then
    MAX_EPOCH=8
fi
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu"

#BITEXT
TEXT=$DATA_DIR/data-bin/split-data-bin80/
DATA_BIN=$TEXT/data-bin0:$TEXT/data-bin1:$TEXT/data-bin2:$TEXT/data-bin3:$TEXT/data-bin4:$TEXT/data-bin5:$TEXT/data-bin6:$TEXT/data-bin7:$TEXT/data-bin8:$TEXT/data-bin9:$TEXT/data-bin10:$TEXT/data-bin11:$TEXT/data-bin12:$TEXT/data-bin13:$TEXT/data-bin14:$TEXT/data-bin15:$TEXT/data-bin16:$TEXT/data-bin17:$TEXT/data-bin18:$TEXT/data-bin19:$TEXT/data-bin20:$TEXT/data-bin21:$TEXT/data-bin22:$TEXT/data-bin23:$TEXT/data-bin24:$TEXT/data-bin25:$TEXT/data-bin26:$TEXT/data-bin27:$TEXT/data-bin28:$TEXT/data-bin29:$TEXT/data-bin30:$TEXT/data-bin31:$TEXT/data-bin32:$TEXT/data-bin33:$TEXT/data-bin34:$TEXT/data-bin35:$TEXT/data-bin36:$TEXT/data-bin37:$TEXT/data-bin38:$TEXT/data-bin39:$TEXT/data-bin40:$TEXT/data-bin41:$TEXT/data-bin42:$TEXT/data-bin43:$TEXT/data-bin44:$TEXT/data-bin45:$TEXT/data-bin46:$TEXT/data-bin47:$TEXT/data-bin48:$TEXT/data-bin49:$TEXT/data-bin50:$TEXT/data-bin51:$TEXT/data-bin52:$TEXT/data-bin53:$TEXT/data-bin54:$TEXT/data-bin55:$TEXT/data-bin56:$TEXT/data-bin57:$TEXT/data-bin58:$TEXT/data-bin59:$TEXT/data-bin60:$TEXT/data-bin61:$TEXT/data-bin62:$TEXT/data-bin63:$TEXT/data-bin64:$TEXT/data-bin65:$TEXT/data-bin66:$TEXT/data-bin67:$TEXT/data-bin68:$TEXT/data-bin69:$TEXT/data-bin70:$TEXT/data-bin71:$TEXT/data-bin72:$TEXT/data-bin73:$TEXT/data-bin74:$TEXT/data-bin75:$TEXT/data-bin76:$TEXT/data-bin77:$TEXT/data-bin78:$TEXT/data-bin79

#BT
BT_TEXT=$DATA_DIR/data-bin/bt_split-data-bin80/
BT_DATA_BIN=$BT_TEXT/data-bin0:$BT_TEXT/data-bin1:$BT_TEXT/data-bin2:$BT_TEXT/data-bin3:$BT_TEXT/data-bin4:$BT_TEXT/data-bin5:$BT_TEXT/data-bin6:$BT_TEXT/data-bin7:$BT_TEXT/data-bin8:$BT_TEXT/data-bin9:$BT_TEXT/data-bin10:$BT_TEXT/data-bin11:$BT_TEXT/data-bin12:$BT_TEXT/data-bin13:$BT_TEXT/data-bin14:$BT_TEXT/data-bin15:$BT_TEXT/data-bin16:$BT_TEXT/data-bin17:$BT_TEXT/data-bin18:$BT_TEXT/data-bin19:$BT_TEXT/data-bin20:$BT_TEXT/data-bin21:$BT_TEXT/data-bin22:$BT_TEXT/data-bin23:$BT_TEXT/data-bin24:$BT_TEXT/data-bin25:$BT_TEXT/data-bin26:$BT_TEXT/data-bin27:$BT_TEXT/data-bin28:$BT_TEXT/data-bin29:$BT_TEXT/data-bin30:$BT_TEXT/data-bin31:$BT_TEXT/data-bin32:$BT_TEXT/data-bin33:$BT_TEXT/data-bin34:$BT_TEXT/data-bin35:$BT_TEXT/data-bin36:$BT_TEXT/data-bin37:$BT_TEXT/data-bin38:$BT_TEXT/data-bin39:$BT_TEXT/data-bin40:$BT_TEXT/data-bin41:$BT_TEXT/data-bin42:$BT_TEXT/data-bin43:$BT_TEXT/data-bin44:$BT_TEXT/data-bin45:$BT_TEXT/data-bin46:$BT_TEXT/data-bin47:$BT_TEXT/data-bin48:$BT_TEXT/data-bin49:$BT_TEXT/data-bin50:$BT_TEXT/data-bin51:$BT_TEXT/data-bin52:$BT_TEXT/data-bin53:$BT_TEXT/data-bin54:$BT_TEXT/data-bin55:$BT_TEXT/data-bin56:$BT_TEXT/data-bin57:$BT_TEXT/data-bin58:$BT_TEXT/data-bin59:$BT_TEXT/data-bin60:$BT_TEXT/data-bin61:$BT_TEXT/data-bin62:$BT_TEXT/data-bin63:$BT_TEXT/data-bin64:$BT_TEXT/data-bin65:$BT_TEXT/data-bin66:$BT_TEXT/data-bin67:$BT_TEXT/data-bin68:$BT_TEXT/data-bin69:$BT_TEXT/data-bin70:$BT_TEXT/data-bin71:$BT_TEXT/data-bin72:$BT_TEXT/data-bin73:$BT_TEXT/data-bin74:$BT_TEXT/data-bin75:$BT_TEXT/data-bin76:$BT_TEXT/data-bin77:$BT_TEXT/data-bin78:$BT_TEXT/data-bin79


#PARALLEL BT
PARALLEL_BT_TEXT=$DATA_DIR/data-bin/parallel_bt_split-data-bin80/
PARALLEL_BT_DATA_BIN=$PARALLEL_BT_TEXT/data-bin0:$PARALLEL_BT_TEXT/data-bin1:$PARALLEL_BT_TEXT/data-bin2:$PARALLEL_BT_TEXT/data-bin3:$PARALLEL_BT_TEXT/data-bin4:$PARALLEL_BT_TEXT/data-bin5:$PARALLEL_BT_TEXT/data-bin6:$PARALLEL_BT_TEXT/data-bin7:$PARALLEL_BT_TEXT/data-bin8:$PARALLEL_BT_TEXT/data-bin9:$PARALLEL_BT_TEXT/data-bin10:$PARALLEL_BT_TEXT/data-bin11:$PARALLEL_BT_TEXT/data-bin12:$PARALLEL_BT_TEXT/data-bin13:$PARALLEL_BT_TEXT/data-bin14:$PARALLEL_BT_TEXT/data-bin15:$PARALLEL_BT_TEXT/data-bin16:$PARALLEL_BT_TEXT/data-bin17:$PARALLEL_BT_TEXT/data-bin18:$PARALLEL_BT_TEXT/data-bin19:$PARALLEL_BT_TEXT/data-bin20:$PARALLEL_BT_TEXT/data-bin21:$PARALLEL_BT_TEXT/data-bin22:$PARALLEL_BT_TEXT/data-bin23:$PARALLEL_BT_TEXT/data-bin24:$PARALLEL_BT_TEXT/data-bin25:$PARALLEL_BT_TEXT/data-bin26:$PARALLEL_BT_TEXT/data-bin27:$PARALLEL_BT_TEXT/data-bin28:$PARALLEL_BT_TEXT/data-bin29:$PARALLEL_BT_TEXT/data-bin30:$PARALLEL_BT_TEXT/data-bin31:$PARALLEL_BT_TEXT/data-bin32:$PARALLEL_BT_TEXT/data-bin33:$PARALLEL_BT_TEXT/data-bin34:$PARALLEL_BT_TEXT/data-bin35:$PARALLEL_BT_TEXT/data-bin36:$PARALLEL_BT_TEXT/data-bin37:$PARALLEL_BT_TEXT/data-bin38:$PARALLEL_BT_TEXT/data-bin39:$PARALLEL_BT_TEXT/data-bin40:$PARALLEL_BT_TEXT/data-bin41:$PARALLEL_BT_TEXT/data-bin42:$PARALLEL_BT_TEXT/data-bin43:$PARALLEL_BT_TEXT/data-bin44:$PARALLEL_BT_TEXT/data-bin45:$PARALLEL_BT_TEXT/data-bin46:$PARALLEL_BT_TEXT/data-bin47:$PARALLEL_BT_TEXT/data-bin48:$PARALLEL_BT_TEXT/data-bin49:$PARALLEL_BT_TEXT/data-bin50:$PARALLEL_BT_TEXT/data-bin51:$PARALLEL_BT_TEXT/data-bin52:$PARALLEL_BT_TEXT/data-bin53:$PARALLEL_BT_TEXT/data-bin54:$PARALLEL_BT_TEXT/data-bin55:$PARALLEL_BT_TEXT/data-bin56:$PARALLEL_BT_TEXT/data-bin57:$PARALLEL_BT_TEXT/data-bin58:$PARALLEL_BT_TEXT/data-bin59:$PARALLEL_BT_TEXT/data-bin60:$PARALLEL_BT_TEXT/data-bin61:$PARALLEL_BT_TEXT/data-bin62:$PARALLEL_BT_TEXT/data-bin63:$PARALLEL_BT_TEXT/data-bin64:$PARALLEL_BT_TEXT/data-bin65:$PARALLEL_BT_TEXT/data-bin66:$PARALLEL_BT_TEXT/data-bin67:$PARALLEL_BT_TEXT/data-bin68:$PARALLEL_BT_TEXT/data-bin69:$PARALLEL_BT_TEXT/data-bin70:$PARALLEL_BT_TEXT/data-bin71:$PARALLEL_BT_TEXT/data-bin72:$PARALLEL_BT_TEXT/data-bin73:$PARALLEL_BT_TEXT/data-bin74:$PARALLEL_BT_TEXT/data-bin75:$PARALLEL_BT_TEXT/data-bin76:$PARALLEL_BT_TEXT/data-bin77:$PARALLEL_BT_TEXT/data-bin78:$PARALLEL_BT_TEXT/data-bin79

#LANG PAIRS
LANG_PAIRS=$DATA_DIR/lang_pairs/lang_pairs.txt
BT_LANG_PAIRS=$DATA_DIR/lang_pairs/bt_lang_pairs.txt
PARALLEL_BT_LANG_PAIRS=$DATA_DIR/lang_pairs/parallel_bt_lang_pairs.txt



GPUS=8
if [ ! $NODES ]; then
    NODES=4
fi
if [ ! $STRATEGY ]; then
    STRATEGY="noise"
fi
if [ $STRATEGY == "noise" ]; then
    python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py ${DATA_BIN} \
        --task summarization_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
        --lang-pairs $LANG_PAIRS --langtoks '{"main":("tgt",None),"bt":("tgt",None),"parallel_bt":("tgt",None)}' --langs $LANGS \
        --extra-data "{\"bt\":\"${BT_DATA_BIN}\",\"parallel_bt\":\"${PARALLEL_BT_DATA_BIN}\"}" \
        --extra-lang-pairs "{\"bt\":\"${BT_LANG_PAIRS}\",\"parallel_bt\":\"${PARALLEL_BT_LANG_PAIRS}\"}" \
        --data-param-list-sampling-ratios '{"main":0.7,"bt":0.2,"parallel_bt":0.1}' \
        --enable-reservsed-directions-shared-datasets \
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
        --rel-pos-buckets 32 --max-rel-pos 128 ${EXTRA_CMDS} 2>&1 | tee -a ${OUTPUT_PATH}/all.log
else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --nnodes=$NODES --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py ${DATA_BIN} \
        --task summarization_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
        --lang-pairs $LANG_PAIRS --langtoks '{"main":("tgt",None),"bt":("tgt",None)}' --langs $LANGS \
        --extra-data "{\"bt\":\"${BT_DATA_BIN}\"}" \
        --extra-lang-pairs "{\"bt\":\"${BT_LANG_PAIRS}\"}" \
        --data-param-list-sampling-ratios '{"main":0.7,"bt":0.2,"parallel_bt":0.1}' \
        --enable-reservsed-directions-shared-datasets \
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
        --rel-pos-buckets 32 --max-rel-pos 128 ${EXTRA_CMDS} 2>&1 | tee -a ${OUTPUT_PATH}/all.log
fi

