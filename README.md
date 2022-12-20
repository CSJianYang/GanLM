GanLM is a sequence-to-sequence pre-training model for both language generation and understanding tasks.


# Fine-tuning on Generation Task
* Abstractive Text Summarization Xsum dataset
``` python
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

if [ ! $MAX_EPOCH ]; then
    MAX_EPOCH=10
fi
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH

python train.py $DATA_DIR \
    --finetune-from-model $PRETRAINED_MODEL_PATH \
    --max-tokens $MAX_TOKENS \
    --tensorboard-logdir $OUTPUT_PATH/tb-log/ --log-file $OUTPUT_PATH/train_log.txt \
    --max-source-positions 720 --max-target-positions 48 \
    --task seq2seq_generation \
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
    --save-interval-updates 100000 --start-save-epoch 4 \
    --fp16 --fp16-init-scale 4 --ddp-backend=no_c10d \
    --rel-pos-buckets 32 --max-rel-pos 128 ${EXTRA_CMDS}
```
# Fine-tuning on Understanding Task 
* XNLI-translation-train-all
``` python
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
EXTRA_CMDS=${16}
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
    --train-langs "en,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh" --eval-langs "en,ar,bg,de,el,es,fr,hi,ru,sw,th,tr,ur,vi,zh" \
    --task xnli --criterion sentence_prediction --arch ${ARCH} --log-interval 100 --log-format "simple" \
    --save-interval-updates 1000000 --keep-interval-updates 1 --save-interval 1 --keep-best-checkpoints 5 \
    --init-token 0 --separator-token 2 --num-classes 3 --add-prev-output-tokens --no-last-checkpoints \
    --dropout 0.1 --attention-dropout 0.1 --pooler-dropout 0.1 --weight-decay ${WEIGHT_DECAY} --clip-norm ${CLIP_NORM} \
    --optimizer adam --adam-betas "(0.9,0.98)" --lr-scheduler inverse_sqrt --lr ${LR} \
    --rel-pos-buckets 32 --max-rel-pos 128 --initialization-strategy ${INITIALIZATION_STRATEGY} \
    --warmup-updates ${WARMUP_STEPS} --max-epoch ${MAX_EPOCH} --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --ddp-backend=no_c10d --fp16 --log-file $OUTPUT_PATH/train.log 2>&1 | tee -a $OUTPUT_PATH/all.log
```



# License
GanLM is MIT-licensed.


# Citation

Please cite as:

``` bibtex
@inproceedings{ganlm,
  title = {xxx},
  author = {xxx},
  booktitle = {xxx},
  year = {2022},
}
```
