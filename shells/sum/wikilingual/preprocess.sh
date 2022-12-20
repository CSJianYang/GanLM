PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
PKL2TXT=/home/v-jiaya/unilm-moe/unilm-moe/scripts/
MAIN_PATH=/home/v-jiaya/unilm-moe/data/wikilingual/download-split/
PRETRAINED_MODEL=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/
SPM_MODEL=$PRETRAINED_MODEL/spm.model
DICT=$PRETRAINED_MODEL/dict.txt
if [ ! -d $MAIN_PATH/Wikilingual/ ]; then
    tar -xvf $MAIN_PATH/WikiLingua-20210602T072436Z-001.zip -C $MAIN_PATH
    #https://drive.google.com/u/0/uc?id=1PM7GFCy2gJL1WHqQz1dzqIDIEN6kfRoi&export=download
fi

DOWNLOAD=$MAIN_PATH/Wikilingual/WikiLingua_data_splits/
TRAIN_SPM=$MAIN_PATH/train_spm/
TRAIN_RAW=$MAIN_PATH/train_raw/
DATA_BIN=$MAIN_PATH/data-bin/
mkdir -p $TRAIN_SPM $TRAIN_RAW
LANGS=(english spanish portuguese french german russian italian indonesian dutch arabic vietnamese chinese thai japanese korean hindi czech turkish)
SIMPLIFIED_LANGS=(en es pt fr de ru it id nl ar vi zh th ja ko hi cs tr)


for i in ${!LANGS[@]}; do
    echo "${LANGS[$i]} | ${SIMPLIFIED_LANGS[$i]}"
    for splt in valid test train; do
        if [ $splt == "valid" ]; then
            echo "Copying $DOWNLOAD/${LANGS[$i]}/${splt}.src.${LANGS} -> $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_src"
            cp $DOWNLOAD/${LANGS[$i]}/val.src.${SIMPLIFIED_LANGS[$i]} $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_src
            echo "Copying $DOWNLOAD/${LANGS[$i]}/${splt}.tgt.${LANGS} -> $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_tgt"
            cp $DOWNLOAD/${LANGS[$i]}/val.tgt.${SIMPLIFIED_LANGS[$i]} $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_tgt
        else
            echo "Copying $DOWNLOAD/${LANGS[$i]}/${splt}.src.${LANGS} -> $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_src"
            cp $DOWNLOAD/${LANGS[$i]}/${splt}.src.${SIMPLIFIED_LANGS[$i]} $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_src
            echo "Copying $DOWNLOAD/${LANGS[$i]}/${splt}.tgt.${LANGS} -> $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_tgt"
            cp $DOWNLOAD/${LANGS[$i]}/${splt}.tgt.${SIMPLIFIED_LANGS[$i]} $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_tgt
        fi
        for data_type in src tgt; do
            cat $TRAIN_RAW/${splt}.${SIMPLIFIED_LANGS[$i]}_${data_type} | spm_encode --model=$SPM_MODEL --output_format=piece > $TRAIN_SPM/${splt}.${SIMPLIFIED_LANGS[$i]}_${data_type}
        done
    done
    echo "Start binarizing $TRAIN/train.${src}-${tgt}..."    
    $PYTHON /home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py  \
        --trainpref $TRAIN_SPM/train --validpref $TRAIN_SPM/valid --testpref $TRAIN_SPM/test \
        --source-lang ${SIMPLIFIED_LANGS[$i]}_src --target-lang ${SIMPLIFIED_LANGS[$i]}_tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 10
done


echo "Copying All Dictionaries $PRETRAINED_MODEL/dict.* -> $DATA_BIN"
cp $PRETRAINED_MODEL/dict.* $DATA_BIN