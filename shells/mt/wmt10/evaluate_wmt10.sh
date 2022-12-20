export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0
DICT=/mnt/msranlp/shumma/data/deltalm/dict.txt
TEXT=/mnt/output/wmt10/test-set/spm-data/
SPM_MODEL=$TEXT/sentencepiece.bpe.model
OUTPUT_PATH=${1}
CHECKPOINT_NAME=${2}
bsz=${3}
beam=${4}
lenpen=${5}
src=${6}
tgt=${7}
BLEU_DIR=${8}
CHECKPOINT_FILE=${OUTPUT_PATH}/${CHECKPOINT_NAME}
if [ ! $lenpen ]; then
    lenpen=1.0
fi

if [ ! $BLEU_DIR ]; then
    BLEU_DIR=/mnt/output/wmt10/BLEU/
fi
if [ "$src" == "hi" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-hi.hi
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-hi.en
elif [ "$src" == "ro" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-ro.ro
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-ro.en
elif [ "$src" == "lv" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-lv.lv
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-lv.en
elif [ "$src" == "et" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-et.et
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-et.en
elif [ "$src" == "gu" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.gu-en.gu
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.gu-en.en
elif [ "$src" == "fr" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-fr.fr
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-fr.en
elif [ "$src" == "cs" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-cs.cs
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-cs.en
elif [ "$src" == "de" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-de.de
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-de.en
elif [ "$src" == "fi" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-fi.fi
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-fi.en
    lenpen=1.0
elif [ "$src" == "tr" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-tr.tr
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-tr.en
elif [ "$tgt" == "hi" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-hi.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-hi.hi
elif [ "$tgt" == "ro" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-ro.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-ro.ro
elif [ "$tgt" == "lv" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-lv.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-lv.lv
elif [ "$tgt" == "et" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-et.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-et.et
elif [ "$tgt" == "gu" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-gu.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-gu.gu
elif [ "$tgt" == "fr" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-fr.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-fr.fr
elif [ "$tgt" == "cs" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-cs.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-cs.cs
elif [ "$tgt" == "de" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-de.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-de.de
elif [ "$tgt" == "fi" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-fi.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-fi.fi
elif [ "$tgt" == "tr" ]; then
    INPUT=/mnt/output/wmt10/test-set/spm-data/test.en-tr.en
    REFERENCE=/mnt/output/wmt10/test-set/raw-data/test.en-tr.tr
else
    echo "Error Language !"
    exit
fi

LANGS="en,fr,cs,de,fi,lv,et,ro,hi,tr,gu"
LANG_PAIRS="en-fr,fr-en,en-cs,cs-en,en-de,de-en,en-fi,fi-en,en-lv,lv-en,en-et,et-en,en-ro,ro-en,en-hi,hi-en,en-tr,tr-en,en-gu,gu-en"
cat $INPUT | python interactive.py $TEXT \
    --path $CHECKPOINT_FILE \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' \
    --task summarization_multi_simple_epoch --langs $LANGS --lang-pairs $LANG_PAIRS \
    --no-repeat-ngram-size 3 --truncate-source --max-source-positions 256 \
    --source-lang ${src} --target-lang ${tgt} --fixed-dictionary $DICT \
    --buffer-size 10000 --batch-size $bsz --beam $beam --lenpen $lenpen \
    --remove-bpe=sentencepiece --no-progress-bar --fp16 > $OUTPUT_PATH/log.txt


OUTPUT_FILE=$CHECKPOINT_NAME.beam${beam}-lenpen${lenpen}-minlen${minlen}
grep ^H $OUTPUT_PATH/log.txt | cut -f3 > $OUTPUT_PATH/$OUTPUT_FILE


mkdir -p $BLEU_DIR
echo "Saving BLEU to $BLEU_DIR/${src}-${tgt}.BLEU..."
echo "$CHECKPOINT_FILE" | tee -a $BLEU_DIR/BLEU.${src}-${tgt}
cat $OUTPUT_PATH/$OUTPUT_FILE | sacrebleu -l $src-$tgt $REFERENCE | tee -a $BLEU_DIR/BLEU.${src}-${tgt}