set -ex
OUTPUT_PATH=${1}
CHECKPOINT_NAME=${2}
bsz=${3}
beam=${4}
lenpen=${5}
minlen=${6}
maxlen=${7}
SPLIT=${8}
lang=${9}
ROUGE_DIR=${10}
CHECKPOINT_FILE=$OUTPUT_PATH/$CHECKPOINT_NAME
if [ ! $SPLIT ]; then
    SPLIT=test
fi

DICT=/mnt/msranlp/shumma/data/deltalm/dict.txt
INPUT_FILE=/mnt/output/wikilingual/download-split/train_spm/${SPLIT}.${lang}_src
DATA=/mnt/output/wikilingual/download-split/data-bin/
REFERENCE=/mnt/output/wikilingual/download-split/train_raw/${SPLIT}.${lang}_tgt

mkdir -p $OUTPUT_PATH

if [ ! $bsz ]; then
    bsz=16
fi
if [ ! $beam ]; then
    beam=8
fi
if [ ! $lenpen ]; then
    lenpen=1.0
fi
if [ ! $minlen ]; then
    minlen=10
fi
if [ ! $maxlen ]; then
    maxlen=58
fi
LANGS="ar,cs,de,en,es,fr,hi,it,id,ja,ko,nl,pt,ru,th,tr,vi,zh"
LANG_PAIRS="ar-ar,cs-cs,de-de,en-en,es-es,fr-fr,hi-hi,it-it,id-id,ja-ja,ko-ko,nl-nl,pt-pt,ru-ru,th-th,tr-tr,vi-vi,zh-zh"
cat $INPUT_FILE | python interactive.py $DATA --fixed-dictionary $DICT \
      --encoder-langtok "tgt" --langtoks '{"main":("tgt", None)}' \
      --task summarization_multi_simple_epoch --source-lang ${lang} --target-lang ${lang} --truncate-source \
      --langs $LANGS --lang-pairs $LANG_PAIRS \
      --path $CHECKPOINT_FILE \
      --buffer-size 100000 --batch-size $bsz \
      --remove-bpe=sentencepiece \
      --beam $beam --lenpen $lenpen --min-len $minlen --max-len-b $maxlen \
      --no-repeat-ngram-size 3 --truncate-source --max-source-positions 512 \
      --fp16 ${extra_args} \
    > $OUTPUT_PATH/log.txt

OUTPUT_FILE=$CHECKPOINT_NAME.beam${beam}-lenpen${lenpen}-minlen${minlen}
grep ^H $OUTPUT_PATH/log.txt | cut -f3 > $OUTPUT_PATH/$OUTPUT_FILE
cat $OUTPUT_PATH/$OUTPUT_FILE | sacrebleu -l ${lang}-${lang} $REFERENCE


TOKENIZER=/mnt/output/mosesdecoder/scripts/tokenizer/tokenizer.perl
CHAR_TOKENIZER=./scripts/wikilingual/tokenizer.py
N_THREADS=20
if [ "${lang}" == "zh" -o "${lang}" == "ja" ]; then
    python $CHAR_TOKENIZER -input $OUTPUT_PATH/$OUTPUT_FILE -output $OUTPUT_PATH/$OUTPUT_FILE.tok -lang ${lang}
    python $CHAR_TOKENIZER -input $REFERENCE -output $REFERENCE.tok -lang ${lang}
else
    cat $OUTPUT_PATH/$OUTPUT_FILE | $TOKENIZER -l ${lang} -no-escape -threads $N_THREADS > $OUTPUT_PATH/$OUTPUT_FILE.tok
    cat $REFERENCE | $TOKENIZER -l ${lang} -no-escape -threads $N_THREADS > $REFERENCE.tok
fi 

#python evaluation/xsum.py --pred $OUTPUT_PATH/$OUTPUT_FILE --gold $REFERENCE --split ${SPLIT}
#ROUGE_DIR=/mnt/output/wikilingual/model/ROUGE/
STR2ID=./scripts/str2id.py
python $STR2ID -generation-str $OUTPUT_PATH/$OUTPUT_FILE.tok -reference-str $REFERENCE.tok -generation-id $OUTPUT_PATH/$OUTPUT_FILE.id -reference-id $REFERENCE.id
mkdir -p $ROUGE_DIR
echo "$OUTPUT_PATH/$OUTPUT_FILE | beam: $beam | min_len: $min_len | max_len: $max_len | len_pen: $lenpen" | tee -a $ROUGE_DIR/ROUGE.${lang}
files2rouge $REFERENCE.id $OUTPUT_PATH/$OUTPUT_FILE.id --ignore_empty_summary | tee -a $ROUGE_DIR/ROUGE.${lang}
#files2rouge /mnt/msranlp/shumma/xsum/raw_data/en/test.tgt /mnt/output/xsum/model/electra-encoder-decoder-v5/lr5e-4-bsz8192-ws10000-wd0.01-dw10_1000_-1-iw1.0_1000_-1-g12d6--share-generator-discriminator/checkpoint_1_150000-ft/lr1e-4-gpus8-uf1-bs4096-sd1-ws1000/generator/checkpoint11.pt.beam8-lenpen1.0-minlen10.post -a "-c 95 -r 1000 -n 2 -a" --ignore_empty_summary 
#-a "-c 95 -m -r 1000 -n 2 -a"
#--ignore_empty_summary