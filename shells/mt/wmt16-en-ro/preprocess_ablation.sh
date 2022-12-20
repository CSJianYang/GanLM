SRC=en
TGT=ro
DICT=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/data-bin/dict.en.txt
RATES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for rate in ${RATES[@]}; do
    TEST=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/test-spm/
    TRAIN=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/ablation/${rate}/train/
    DATA_BIN=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/ablation/${rate}/data-bin/
    echo "Start binarizing $TRAIN/train.${SRC}-${TGT}..."    
    /home/v-jiaya/miniconda3/envs/amlt8/bin/python /home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py  \
        --trainpref $TRAIN/train.${SRC}-${TGT} --validpref $TEST/valid.${SRC}-${TGT} \
        --source-lang $SRC --target-lang $TGT \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 20
done