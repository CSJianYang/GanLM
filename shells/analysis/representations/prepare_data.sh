OLD_SPM_DATA=/home/v-jiaya/unilm-moe/data/representations/similarity/
RAW_DATA=/home/v-jiaya/unilm-moe/data/representations/raw_data/
SPM_DATA=/home/v-jiaya/unilm-moe/data/representations/spm_data/
SPM_MODEL=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/spm.model
mkdir -p $SPM_DATA $RAW_DATA
LANGS=(en fr cs de fi lv et ro hi tr gu)
for lg in ${LANGS[@]}; do
    echo "${lg}"
    cat $OLD_SPM_DATA/parallel.${lg} | python /home/v-jiaya/unilm-moe/unilm-moe/scripts/spm_decode.py > $RAW_DATA/parallel.${lg}
    cat $RAW_DATA/parallel.${lg} | spm_encode --model=$SPM_MODEL --output_format=piece > $SPM_DATA/parallel.${lg}
done