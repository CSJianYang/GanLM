PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
DICT=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/dict.txt
LANGS=(en ru)
DATA=/home/v-jiaya/unilm-moe/data/webnlg/raw_data/

for lg in ${LANGS[@]}; do
    mkdir -p $DATA/${lg}/spm_data/
    cp $DATA/${lg}_sp/* $DATA/${lg}/spm_data/
    SPM_DATA=$DATA/${lg}/spm_data/
    DATA_BIN=$DATA/${lg}/data-bin/
    echo "Start binarizing $TRAIN/train.src-tgt..."    
    $PYTHON /home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py  \
        --trainpref $SPM_DATA/train --validpref $SPM_DATA/valid --testpref $SPM_DATA/test \
        --source-lang src --target-lang tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 10
done
