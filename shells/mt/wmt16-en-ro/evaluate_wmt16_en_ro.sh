set -ex
OUTPUT_PATH=${1}
CHECKPOINT_NAME=${2}
bsz=${3}
beam=${4}
lenpen=${5}
minlen=${6}
maxlen=${7}
SPLIT=${8}
src=${9}
tgt=${10}
CHECKPOINT_FILE=${OUTPUT_PATH}/${CHECKPOINT_NAME}

INPUT_FILE=/mnt/output/wmt16-en-ro/test-spm/${SPLIT}.en-ro.${src}
DATA=/mnt/output/wmt16-en-ro/data-bin/
REFERENCE=/mnt/output/wmt16-en-ro/test-raw/${SPLIT}.en-ro.${tgt}

# moses
MOSES=/mnt/output/mosesdecoder/scripts/
REPLACE_UNICODE_PUNCT=$MOSES/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/tokenizer/tokenizer.perl
N_THREADS=20
if [ ! -d $SCRIPTS ]; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git /mnt/output/mosesdecoder/
fi


# Sennrich's WMT16 scripts for Romanian preprocessing
WMT16_SCRIPTS=./scripts/wmt16-en-ro/preprocess/
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/remove-diacritics.py


cat $INPUT_FILE | python interactive.py $DATA \
      --task seq2seq_generation --source-lang ${src} --target-lang ${tgt} \
      --path $CHECKPOINT_FILE \
      --buffer-size 100000 --batch-size $bsz \
      --remove-bpe=sentencepiece \
      --beam $beam --lenpen $lenpen --max-len-b ${maxlen} --min-len $minlen --no-repeat-ngram-size 3 --truncate-source \
      --fp16 ${extra_args} \
    > $OUTPUT_PATH/log.txt
    
OUTPUT_FILE=$CHECKPOINT_NAME.beam${beam}-lenpen${lenpen}-minlen${minlen}
grep ^H $OUTPUT_PATH/log.txt | cut -f3 > $OUTPUT_PATH/$OUTPUT_FILE

if [ "$tgt" == "ro" ]; then 
    PREPROCESSING="${REPLACE_UNICODE_PUNCT} | ${NORM_PUNC} -l ${tgt:0:2} | ${REM_NON_PRINT_CHAR} | python ${NORMALIZE_ROMANIAN} | python ${REMOVE_DIACRITICS} | ${TOKENIZER} -l ${tgt:0:2} -no-escape -threads ${N_THREADS}"
else
    PREPROCESSING="${REPLACE_UNICODE_PUNCT} | ${NORM_PUNC} -l ${tgt:0:2} | ${REM_NON_PRINT_CHAR} | ${TOKENIZER} -l ${tgt:0:2} -no-escape -threads $N_THREADS"
fi
echo "PREPROCESS CMD: $PREPROCESSING"
cat $OUTPUT_PATH/$OUTPUT_FILE | eval $PREPROCESSING > $OUTPUT_PATH/$OUTPUT_FILE.tok.post
cat $REFERENCE | eval $PREPROCESSING > $REFERENCE.tok.post
cat $OUTPUT_PATH/$OUTPUT_FILE.tok.post | sacrebleu -tok 'none' -s 'none' $REFERENCE.tok.post