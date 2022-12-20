PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
DICT=/mnt/msranlp/shumma/data/16g/dict.txt
SPM_MODEL=/mnt/msranlp/shumma/data/16g/sentencepiece.bpe.model
RAW_DIR=/home/v-jiaya/unilm-moe/data/cnn_dm/raw_data/
SPM_DIR=/home/v-jiaya/unilm-moe/data/cnn_dm/spm_data/
BINARY_DIR=/home/v-jiaya/unilm-moe/data/cnn_dm/data-bin/
for split in valid test train; do
    INPUT=$RAW_DIR/$split.src
    OUTPUT=$SPM_DIR/$split.src
    echo "$INPUT -> $OUTPUT"
    cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
    INPUT=$RAW_DIR/$split.tgt
    OUTPUT=$SPM_DIR/$split.tgt
    echo "$INPUT -> $OUTPUT"
    cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
done

$PYTHON /home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py  \
    --trainpref $SPM_DIR/train --validpref $SPM_DIR/valid --testpref $SPM_DIR/test \
    --source-lang src --target-lang tgt \
    --destdir $BINARY_DIR \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 20