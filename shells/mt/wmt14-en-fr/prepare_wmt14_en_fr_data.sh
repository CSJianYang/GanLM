#!/bin/bash
PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
SPM_MODEL=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/spm.model
DICT=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/dict.txt


URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v9.tgz"
    "training-giga-fren.tar"
    "dev.tgz"
    "test-full.tgz"
)
CORPORA=(
    "training/europarl-v7.fr-en"
    "commoncrawl.fr-en"
    "un/undoc.2000.fr-en"
    "training/news-commentary-v9.fr-en"
    "giga-fren.release2.fixed"
)
OUTPUT_DIR=/home/v-jiaya/unilm-moe/data/wmt14-en-fr/
src=en
tgt=fr
ORIG_DATA=$OUTPUT_DIR/orig_data/
RAW_DATA=$OUTPUT_DIR/raw_data/
SPM_DATA=$OUTPUT_DIR/spm_data/
DATA_BIN=$OUTPUT_DIR/data-bin/
mkdir -p $ORIG_DATA $RAW_DATA $SPM_DATA

cd $ORIG_DATA
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
#gunzip giga-fren.release2.fixed.*.gz
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    if [ ! -f $RAW_DATA/train.$l ]; then
        for f in "${CORPORA[@]}"; do
            cat $ORIG_DATA/$f.$l >> $RAW_DATA/train.$l
        done
    fi
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $ORIG_DATA/dev/newstest2013-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\¡¯/\'/g" > $RAW_DATA/valid.$l
        
    grep '<seg id' $ORIG_DATA/test-full/newstest2014-fren-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $RAW_DATA/test.$l
done

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        if [ ! -f $SPM_DATA/$f ]; then
            echo "SPM $RAW_DATA/$f -> $SPM_DATA/$f"
            cat $RAW_DATA/$f | spm_encode --model=$SPM_MODEL --output_format=piece > $SPM_DATA/$f
        else
            echo "Skipping SPM Preprocess..."
        fi
    done
done


$PYTHON /home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py  \
        --trainpref $SPM_DATA/train --validpref $SPM_DATA/valid --testpref $SPM_DATA/test \
        --source-lang $src --target-lang $tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 10
