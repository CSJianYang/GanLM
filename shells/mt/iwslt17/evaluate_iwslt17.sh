export CUDA_VISIBLE_DEVICES=0
TEXT=/mnt/output/test-set/spm-data/
DICT=/mnt/msranlp/shumma/data/deltalm/dict.txt
SPM_MODEL=$TEXT/sentencepiece.bpe.model
OUTPUT_PATH=${1}
CHECKPOINT_NAME=${2}
bsz=${3}
beam=${4}
lenpen=${5}
src=${6}
tgt=${7}
BLEU_DIR=${8}
SPLIT=${9}
MODEL=${OUTPUT_PATH}/${CHECKPOINT_NAME}
if [ ! $SPLIT ]; then
    SPLIT=test
fi
if [ ${src} = ${tgt} ]; then
    echo "Same Source and Target Language ${src}/${tgt}"
    exit
fi
if [ ! $lenpen ]; then
    lenpen=1.0
fi
if [ ! $BLEU_DIR ]; then
    BLEU_DIR=/mnt/output/iwslt17/BLEU/
fi
mkdir -p $BLEU_DIR

INPUT_FILE=/mnt/output/iwslt17/spm_data/${SPLIT}.${src}-${tgt}.${src}
DATA=/mnt/output/iwslt17/data-bin/
REFERENCE=/mnt/output/iwslt17/raw_data/${SPLIT}.${src}-${tgt}.${tgt}


LANGS="de,en,it,nl,ro"
LANG_PAIRS="en-de,de-en,en-it,it-en,en-nl,nl-en,en-ro,ro-en"
cat $INPUT_FILE | python interactive.py $DATA \
    --path $MODEL \
    --langtoks '{"main":("tgt",None)}' --fixed-dictionary $DICT \
    --task summarization_multi_simple_epoch \
    --langs $LANGS --lang-pairs $LANG_PAIRS \
    --source-lang $src --target-lang $tgt \
    --buffer-size 10000 --batch-size $bsz --beam $beam --lenpen $lenpen \
    --remove-bpe=sentencepiece --no-progress-bar --fp16 > $OUTPUT_PATH/log.txt


OUTPUT_FILE=$CHECKPOINT_NAME.${src}2${tgt}.beam${beam}-lenpen${lenpen}-minlen${minlen}
echo "Saving BLEU to $BLEU_DIR/BLEU.${src}-${tgt}..."
echo "$MODEL" | tee -a $BLEU_DIR/BLEU.${src}-${tgt}
cat $OUTPUT_PATH/log.txt | grep -P "^H" | cut -f 3- > $OUTPUT_PATH/$OUTPUT_FILE
cat $OUTPUT_PATH/$OUTPUT_FILE | sacrebleu -l $src-$tgt $REFERENCE | tee -a $BLEU_DIR/BLEU.${src}-${tgt}