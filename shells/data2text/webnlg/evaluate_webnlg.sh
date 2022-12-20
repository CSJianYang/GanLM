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

CHECKPOINT_FILE=$OUTPUT_PATH/$CHECKPOINT_NAME
if [ ! $SPLIT ]; then
    SPLIT=test
fi

INPUT_FILE=/mnt/output/webnlg/raw_data/${lang}/spm_data/${SPLIT}.src
DATA=/mnt/output/webnlg/raw_data/${lang}/data-bin/
REFERENCE=/mnt/output/GEM-metrics/test_data/webnlg_${lang}_${SPLIT}.json

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
cat $INPUT_FILE | python interactive.py $DATA \
      --task seq2seq_generation --source-lang src --target-lang tgt \
      --path $CHECKPOINT_FILE \
      --buffer-size 100000 --batch-size $bsz \
      --remove-bpe=sentencepiece \
      --beam $beam --lenpen $lenpen --max-len-b $maxlen --min-len $minlen --no-repeat-ngram-size 3 --truncate-source --max-source-positions 608 \
      --fp16 ${extra_args} \
    > $OUTPUT_PATH/log.txt

OUTPUT_FILE=$CHECKPOINT_NAME.beam${beam}-lenpen${lenpen}-minlen${minlen}
grep ^H $OUTPUT_PATH/log.txt | cut -f3 > $OUTPUT_PATH/$OUTPUT_FILE.$lang

ROUGE_DIR=/mnt/output/webnlg/model/ROUGE/
mkdir -p $ROUGE_DIR
echo "$OUTPUT_PATH/$OUTPUT_FILE | beam: $beam | min_len: $min_len | max_len: $max_len | len_pen: $lenpen" | tee -a $ROUGE_DIR/ROUGE.txt
python /mnt/output/GEM-metrics/wrap-outputs.py $lang $OUTPUT_PATH/$OUTPUT_FILE.$lang $OUTPUT_PATH/$OUTPUT_FILE.$lang.json
python /mnt/output/GEM-metrics/run_metrics.py -r $REFERENCE $OUTPUT_PATH/$OUTPUT_FILE.$lang.json --metric-list rouge
#/home/v-jiaya/miniconda3/envs/amlt8/bin/python /mnt/output/GEM-metrics/wrap-outputs.py en /mnt/output/webnlg/model/en/electra-encoder-decoder-v6/lr3e-4-bsz8192-ws10000-wd0.05-dw10_-1_-1-iw1.0_-1_-1-g12d4/checkpoint_1_290000-ft/lr1e-4-gpus8-uf2-bs2048-sd1-ws4000-wd0.05/both-5-1.0//checkpoint_last.pt.beam8-lenpen1.0-minlen0.en /mnt/output/webnlg/model/en/electra-encoder-decoder-v6/lr3e-4-bsz8192-ws10000-wd0.05-dw10_-1_-1-iw1.0_-1_-1-g12d4/checkpoint_1_290000-ft/lr1e-4-gpus8-uf2-bs2048-sd1-ws4000-wd0.05/both-5-1.0//checkpoint_last.pt.beam8-lenpen1.0-minlen0.en.json
#/home/v-jiaya/miniconda3/envs/amlt8/bin/python /mnt/output/GEM-metrics/run_metrics.py -r /mnt/output/GEM-metrics/test_data/webnlg_en_validation.json /mnt/output/webnlg/model/en/electra-encoder-decoder-v6/lr3e-4-bsz8192-ws10000-wd0.05-dw10_-1_-1-iw1.0_-1_-1-g12d4/checkpoint_1_290000-ft/lr1e-4-gpus8-uf2-bs2048-sd1-ws4000-wd0.05/both-5-1.0//checkpoint_last.pt.beam8-lenpen1.0-minlen0.en.json --metric-list rouge


#-a "-c 95 -m -r 1000 -n 2 -a"
#files2rouge /mnt/msranlp/shumma/xsum/raw_data/en/test.tgt /mnt/output/xsum/model/electra-encoder-decoder-v5/lr5e-4-bsz8192-ws10000-wd0.01-dw10_1000_-1-iw1.0_1000_-1-g12d6--share-generator-discriminator/checkpoint_1_150000-ft/lr1e-4-gpus8-uf1-bs4096-sd1-ws1000/generator/checkpoint11.pt.beam8-lenpen1.0-minlen10.post -a "-c 95 -r 1000 -n 2 -a" --ignore_empty_summary 