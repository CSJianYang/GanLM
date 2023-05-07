#!/usr/bin/env bash
set -ex

INPUT_FILE=$1
OUTPUT_PATH=$2
CHECKPOINT_FILE=$3
DATA=$4
REFERENCE=$5
extra_args=$6

mkdir -p $OUTPUT_PATH

cat $INPUT_FILE | python scripts/truncate.py \
    | python interactive.py $DATA \
      --task seq2seq_generation --source-lang src --target-lang tgt \
      --path $CHECKPOINT_FILE \
      --buffer-size 100000 --batch-size 32 \
      --remove-bpe=sentencepiece \
      --beam 6 --lenpen 1.0 --max-len-b 60 --min-len 10 --no-repeat-ngram-size 3 \
      --fp16 ${extra_args} \
    > $OUTPUT_PATH/log.txt
grep ^H $OUTPUT_PATH/log.txt | cut -f3 > $OUTPUT_PATH/output.txt

python evaluation/xsum.py --pred $OUTPUT_PATH/output.txt --gold $REFERENCE --split test