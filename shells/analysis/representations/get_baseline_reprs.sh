LANGS=(en fr cs de fi lv et ro hi tr gu)
PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
DATA=/home/v-jiaya/unilm-moe/data/representations/spm_data/
MODEL=/home/v-jiaya/unilm-moe/data/wmt10/model/transformer/avg3_7.pt
for lg in ${LANGS[@]}; do
    echo "Get ${lg} Representations..."
    $PYTHON /home/v-jiaya/unilm-moe/unilm-moe/scripts/wmt10/get_wmt10_representations.py --src_fn $DATA/parallel.$lg --ckpt_path $MODEL --model_name baseline_3_7 --bsz 32
done