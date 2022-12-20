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
BLEU_DIR=${11}
SCRIPTS=/mnt/output/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DICT=/mnt/output/PretrainedModels/xlmr.base/dict.txt
if [ ! -d $SCRIPTS ]; then
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git /mnt/output/mosesdecoder/
fi


CHECKPOINT_FILE=${OUTPUT_PATH}/${CHECKPOINT_NAME}
INPUT_FILE=/mnt/output/wmt14-en-fr/spm_data/${SPLIT}.${src}
DATA=/mnt/output/wmt14-en-fr/data-bin/
REFERENCE=/mnt/output/wmt14-en-fr/raw_data/${SPLIT}.${tgt}

cat $INPUT_FILE | python interactive.py $DATA \
      --task seq2seq_generation --source-lang ${src} --target-lang ${tgt} \
      --path $CHECKPOINT_FILE \
      --buffer-size 100000 --batch-size $bsz \
      --remove-bpe=sentencepiece \
      --beam $beam --lenpen $lenpen --max-len-b ${maxlen} --min-len ${minlen} --no-repeat-ngram-size 3 --truncate-source \
      --fp16 ${extra_args} \
    > $OUTPUT_PATH/log.txt
    
OUTPUT_FILE=$CHECKPOINT_NAME.beam${beam}-lenpen${lenpen}-minlen${minlen}
grep ^H $OUTPUT_PATH/log.txt | cut -f3 > $OUTPUT_PATH/$OUTPUT_FILE
cat $OUTPUT_PATH/$OUTPUT_FILE | sacrebleu -l ${src}-${tgt} $REFERENCE

mkdir -p ${BLEU_DIR}
cat $OUTPUT_PATH/$OUTPUT_FILE | perl $TOKENIZER -threads 20 -a -l ${tgt} > $OUTPUT_PATH/$OUTPUT_FILE.tok.post #| sed  's/-/ - /g'
cat $REFERENCE | perl $TOKENIZER -threads 20 -a -l ${tgt} > $REFERENCE.tok.post #| sed  's/-/ - /g'
perl ./scripts/bleu.pl $REFERENCE.tok.post < $OUTPUT_PATH/$OUTPUT_FILE.tok.post 
#| ${BLEU_DIR}/BLEU.${src}-${tgt}
#perl ./scripts/bleu.pl /mnt/output/wmt14-en-de/raw_data/test.de.tok.post < /mnt/output/wmt14-en-de/model/electra-encoder-decoder-v6/lr3e-4-bsz8192-ws10000-wd0.05-dw10_-1_-1-iw1.0_-1_-1-g12d4-125K/checkpoint_1_125000-ft/lr1e-4-gpus8-uf32-bs2048-sd1-ws1000-wd0.05/both-5-1.0//checkpoint_best.pt.beam8-lenpen1.0-minlen0.tok.post