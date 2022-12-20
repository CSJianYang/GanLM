#please downlod training and valid data from https://wit3.fbk.eu/mt.php?release=2017-01-trnmted
#Please download test tst2017 files from https://wit3.fbk.eu/mt.php?release=2017-01-mted-test
PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
SPM_MODEL=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/spm.model
DICT=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/dict.txt
BINARIZE=/home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py
DATA=/home/v-jiaya/unilm-moe/data/iwslt17/
ORIG_DATA=$DATA/DeEnItNlRo-DeEnItNlRo/
RAW_DATA=$DATA/raw_data/
SPM_DATA=$DATA/spm_data/
DATA_BIN=$DATA/data-bin/
TST=$DATA/tst2017/
mkdir -p $ORIG_DATA $RAW_DATA $SPM_DATA
SRCS=(
    "en"
    "de"
    "nl"
    "it"
    "ro"
)
TGTS=(
    "en"
    "de"
    "nl"
    "it"
    "ro"
)


action=$1

if [ "$action" == "download" -o "$action" == "all" ]; then
    cd $TST
    echo "Download train data..."
    wget https://wit3.fbk.eu/archive/2017-01-trnmted//texts/DeEnItNlRo/DeEnItNlRo/DeEnItNlRo-DeEnItNlRo.tgz -P $DATA/
    tar -xvf $DATA/DeEnItNlRo-DeEnItNlRo.tar -C $DATA/
    echo "Download test data..."
    for SRC in "${SRCS[@]}"; do
        for TGT in "${TGTS[@]}"; do
            if [ "${SRC}" != "${TGT}" ]; then
                wget https://wit3.fbk.eu/archive/2017-01-mted-test/texts/${SRC}/${TGT}/${SRC}-${TGT}.tgz -P $TST/
            fi
        done
    done
fi


if [ "$action" == "extract" -o "$action" == "all" ]; then
    echo "pre-processing train valid test data..."
    for SRC in "${SRCS[@]}"; do
        for TGT in "${TGTS[@]}"; do
            if [ "${SRC}" != "${TGT}" ]; then
                echo "pre-processing ${SRC}-${TGT} train valid test data"
                for LANG in "${SRC}" "${TGT}"; do
                    cat $ORIG_DATA/train.tags.${SRC}-${TGT}.${LANG} \
                        | grep -v '<url>' \
                        | grep -v '<talkid>' \
                        | grep -v '<keywords>' \
                        | grep -v '<speaker>' \
                        | grep -v '<reviewer' \
                        | grep -v '<translator' \
                        | grep -v '<doc' \
                        | grep -v '</doc>' \
                        | sed -e 's/<title>//g' \
                        | sed -e 's/<\/title>//g' \
                        | sed -e 's/<description>//g' \
                        | sed -e 's/<\/description>//g' \
                        | sed 's/^\s*//g' \
                        | sed 's/\s*$//g' > $RAW_DATA/train.${SRC}-${TGT}.${LANG}
                    grep '<seg id' $ORIG_DATA/IWSLT17.TED.dev2010.${SRC}-${TGT}.${LANG}.xml \
                        | sed -e 's/<seg id="[0-9]*">\s*//g' \
                        | sed -e 's/\s*<\/seg>\s*//g' \
                        | sed -e "s/\¡¯/\'/g" > $RAW_DATA/valid.${SRC}-${TGT}.${LANG}
                    grep '<seg id' $ORIG_DATA/IWSLT17.TED.tst2010.${SRC}-${TGT}.${LANG}.xml \
                        | sed -e 's/<seg id="[0-9]*">\s*//g' \
                        | sed -e 's/\s*<\/seg>\s*//g' \
                        | sed -e "s/\¡¯/\'/g" >> $RAW_DATA/valid.${SRC}-${TGT}.${LANG}                  
                done
                if [ ! -d $TST/${SRC}-${TGT} ]; then
                    tar zxvf $TST/${SRC}-${TGT}.tgz -C $TST
                fi
                grep '<seg id' "$TST/${SRC}-${TGT}/IWSLT17.TED.tst2017.mltlng.${SRC}-${TGT}.${SRC}.xml" \
                    | sed -e 's/<seg id="[0-9]*">\s*//g' \
                    | sed -e 's/\s*<\/seg>\s*//g' \
                    | sed -e "s/\¡¯/\'/g" > $RAW_DATA/test.${SRC}-${TGT}.${SRC}
                if [ ! -d $TST/${TGT}-${SRC} ]; then
                    tar zxvf $TST/${TGT}-${SRC}.tgz -C $TST
                fi
                grep '<seg id' "$TST/${TGT}-${SRC}/IWSLT17.TED.tst2017.mltlng.${TGT}-${SRC}.${TGT}.xml" \
                    | sed -e 's/<seg id="[0-9]*">\s*//g' \
                    | sed -e 's/\s*<\/seg>\s*//g' \
                    | sed -e "s/\¡¯/\'/g" > $RAW_DATA/test.${SRC}-${TGT}.${TGT}
            fi
        done
    done
fi


if [ "$action" == "spm" -o "$action" == "all" ]; then    
    for SRC in "${SRCS[@]}"; do
        for TGT in "${TGTS[@]}"; do
            if [ "${SRC}" != "${TGT}" ]; then
                for LANG in "${SRC}" "${TGT}"; do
                    echo "SPM ${SRC}-${TGT}.${LANG}..."
                    spm_encode --model=$SPM_MODEL --output_format=piece < $RAW_DATA/train.${SRC}-${TGT}.${LANG} > $SPM_DATA/train.${SRC}-${TGT}.${LANG}
                    spm_encode --model=$SPM_MODEL --output_format=piece < $RAW_DATA/valid.${SRC}-${TGT}.${LANG} > $SPM_DATA/valid.${SRC}-${TGT}.${LANG}
                    spm_encode --model=$SPM_MODEL --output_format=piece < $RAW_DATA/test.${SRC}-${TGT}.${LANG}  > $SPM_DATA/test.${SRC}-${TGT}.${LANG}
                done
            fi
        done
    done
fi


if [ "$action" == "binary" -o "$action" == "all" ]; then
    for SRC in "${SRCS[@]}"; do
        for TGT in "${TGTS[@]}"; do
            if [ "${SRC}" != "${TGT}" ]; then
                echo "binary ${SRC}-${TGT}"
                $PYTHON $BINARIZE --source-lang ${SRC} --target-lang ${TGT} \
                  --trainpref $SPM_DATA/train.${SRC}-${TGT} --validpref $SPM_DATA/valid.${SRC}-${TGT} --testpref $SPM_DATA/test.${SRC}-${TGT} \
                  --destdir $DATA_BIN \
                  --srcdict $DICT \
                  --tgtdict $DICT \
                  --workers 10      
            fi
        done    
    done
fi