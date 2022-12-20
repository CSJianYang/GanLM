MAIN_PATH=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/download/
SRC=en
TGT=ro
NEW_SRC=en_XX
NEW_TGT=ro_RO
SPM_MODEL=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/spm.model
DICT=/home/v-jiaya/unilm-moe/data/PretrainedModels/multilingual/dict.txt
DOWNLOAD=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/download/
TRAIN=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/train/
TRAIN_BT=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/train_bt/
TEST=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/test-spm/
TEST_RAW=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/test-raw/
#DATA_BIN1=/home/v-jiaya/mBART/data/wmt16-En-Ro/download/data-bin-mBART/
DATA_BIN=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/data-bin/
DATA_BIN_BT=/home/v-jiaya/unilm-moe/data/wmt16-en-ro/data-bin_bt/
PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
mkdir -p $DOWNLOAD $TRAIN $TRAIN_BT $TEST $TEST_RAW

# moses
MOSES=/home/v-jiaya/mosesdecoder/
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/tokenizer/scripts/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
CLEAN=$MOSES/scripts/training/clean-corpus-n.perl
N_THREADS=20
if [ ! -d $MOSES ]; then
    git clone git@github.com:moses-smt/mosesdecoder.git
fi


# Sennrich's WMT16 scripts for Romanian preprocessing
WMT16_SCRIPTS=/home/v-jiaya/unilm-moe/unilm-moe/scripts/wmt16-en-ro/
NORMALIZE_ROMANIAN=$WMT16_SCRIPTS/normalise-romanian.py
REMOVE_DIACRITICS=$WMT16_SCRIPTS/remove-diacritics.py
if [ ! -d $WMT16_SCRIPTS ]; then
    git clone git@github.com:rsennrich/wmt16-scripts.git
fi


if [ ! -f $DOWNLOAD/dev.tgz -a ! -f $DOWNLOAD/ro-en.tgz -a ! -f $DOWNLOAD/ro-en.txt.zip -a ! -f $DOWNLOAD/corpus.bt.ro-en.ro.gz ]; then
	wget -c http://data.statmt.org/wmt18/translation-task/dev.tgz -P $MAIN_PATH
	wget http://www.statmt.org/europarl/v7/ro-en.tgz -P $MAIN_PATH
	wget http://opus.lingfil.uu.se/download.php?f=SETIMES2/en-ro.txt.zip -O SETIMES2.ro-en.txt.zip -P $MAIN_PATH
	wget -nc http://data.statmt.org/rsennrich/wmt16_backtranslations/ro-en/corpus.bt.ro-en.en.gz -P $MAIN_PATH
	wget -nc http://data.statmt.org/rsennrich/wmt16_backtranslations/ro-en/corpus.bt.ro-en.ro.gz -P $MAIN_PATH

	echo "Unzip parallel data..."
	tar -xzf dev.tgz
	tar -xf ro-en.tgz
	unzip SETIMES2.ro-en.txt.zip
	gzip -d corpus.bt.ro-en.en.gz corpus.bt.ro-en.ro.gz
fi


SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR |                                            $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"

SRC_PREPROCESSING_WITHOUT_TOKENIZER="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR"
TGT_PREPROCESSING_WITHOUT_TOKENIZER="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $NORMALIZE_ROMANIAN | $REMOVE_DIACRITICS"

# concatenate bilingual data
if [ ! -f $TRAIN/train.all.$SRC-$TGT.$SRC -a ! -f $TRAIN/train.all.$SRC-$TGT.$TGT ]; then
	echo "Concatenating $SRC bilingual data ..."
  cat $DOWNLOAD/europarl-v7.ro-en.$SRC $DOWNLOAD/SETIMES.en-ro.$SRC | spm_encode --model=$SPM_MODEL --output_format=piece > $TRAIN/train.all.$SRC-$TGT.$SRC
  echo "Concatenating $TGT bilingual data ..."
  cat $DOWNLOAD/europarl-v7.ro-en.$TGT $DOWNLOAD/SETIMES.en-ro.$TGT | spm_encode --model=$SPM_MODEL --output_format=piece > $TRAIN/train.all.$SRC-$TGT.$TGT
fi


if [ ! -f $TRAIN_BT/train.all.$SRC-$TGT.$SRC -a ! -f $TRAIN_BT/train.all.$SRC-$TGT.$TGT ]; then
	echo "Concatenating BT $SRC bilingual data ..."
	cat $DOWNLOAD/corpus.bt.ro-en.$SRC $DOWNLOAD/europarl-v7.ro-en.$SRC $DOWNLOAD/SETIMES.en-ro.$SRC | spm_encode --model=$SPM_MODEL --output_format=piece > $TRAIN_BT/train.all.$SRC-$TGT.$SRC
  echo "Concatenating BT $TGT bilingual data ..."
	cat $DOWNLOAD/corpus.bt.ro-en.$TGT $DOWNLOAD/europarl-v7.ro-en.$TGT $DOWNLOAD/SETIMES.en-ro.$TGT | spm_encode --model=$SPM_MODEL --output_format=piece > $TRAIN_BT/train.all.$SRC-$TGT.$TGT
fi



if [ ! -f $TRAIN/train.$SRC-$TGT.$SRC -a ! -f $TRAIN/train.$SRC-$TGT.$TGT ]; then
    perl $CLEAN -ratio 2.0 $TRAIN/train.all.$SRC-$TGT $SRC $TGT $TRAIN/train.$SRC-$TGT 1 1024
fi

if [ ! -f $TRAIN_BT/train.$SRC-$TGT.$SRC -a ! -f $TRAIN_BT/train.$SRC-$TGT.$TGT ]; then
    perl $CLEAN -ratio 2.0 $TRAIN_BT/train.all.$SRC-$TGT $SRC $TGT $TRAIN_BT/train.$SRC-$TGT 1 1024
fi

if [ ! -f $TEST/test.${SRC}-${TGT}.${SRC} -a ! -f $TEST/test.${SRC}-${TGT}.${TGT} ]; then
    #valid
    cat $MAIN_PATH/dev/newsdev2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/valid.${SRC}-${TGT}.${SRC}
    cat $MAIN_PATH/dev/newsdev2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/valid.${SRC}-${TGT}.${TGT}
    #cat $MAIN_PATH/dev/newstest2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/valid.${NEW_SRC}-${NEW_TGT}.${NEW_SRC}
    #cat $MAIN_PATH/dev/newstest2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/valid.${NEW_SRC}-${NEW_TGT}.${NEW_TGT} 
    #test
    cat $MAIN_PATH/dev/newstest2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/test.${SRC}-${TGT}.${SRC}
    cat $MAIN_PATH/dev/newstest2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/test.${SRC}-${TGT}.${TGT}
    #cat $MAIN_PATH/dev/newstest2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/test.${NEW_SRC}-${NEW_TGT}.${NEW_SRC}
    #cat $MAIN_PATH/dev/newstest2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM | spm_encode --model=$SPM_MODEL --output_format=piece > $TEST/test.${NEW_SRC}-${NEW_TGT}.${NEW_TGT}
fi

if [ ! -f $TEST_RAW/test.${SRC}-${TGT}.${SRC} -a ! -f $TEST_RAW/test.${SRC}-${TGT}.${TGT} ]; then
    #valid
    cat $MAIN_PATH/dev/newsdev2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM > $TEST_RAW/valid.${SRC}-${TGT}.${SRC}
    cat $MAIN_PATH/dev/newsdev2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM > $TEST_RAW/valid.${SRC}-${TGT}.${TGT}
    #cat $MAIN_PATH/dev/newstest2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM > $TEST_RAW/valid.${NEW_SRC}-${NEW_TGT}.${NEW_SRC}
    #cat $MAIN_PATH/dev/newstest2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM > $TEST_RAW/valid.${NEW_SRC}-${NEW_TGT}.${NEW_TGT} 
    #test
    cat $MAIN_PATH/dev/newstest2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM > $TEST_RAW/test.${SRC}-${TGT}.${SRC}
    cat $MAIN_PATH/dev/newstest2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM > $TEST_RAW/test.${SRC}-${TGT}.${TGT}
    #cat $MAIN_PATH/dev/newstest2016-enro-src.${SRC}.sgm | $INPUT_FROM_SGM > $TEST_RAW/test.${NEW_SRC}-${NEW_TGT}.${NEW_SRC}
    #cat $MAIN_PATH/dev/newstest2016-enro-ref.${TGT}.sgm | $INPUT_FROM_SGM > $TEST_RAW/test.${NEW_SRC}-${NEW_TGT}.${NEW_TGT}
fi



echo "Start binarizing $TRAIN/train.${SRC}-${TGT}..."    
$PYTHON /home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py  \
    --trainpref $TRAIN/train.${SRC}-${TGT} --validpref $TEST/valid.${SRC}-${TGT} \
    --source-lang $SRC --target-lang $TGT \
    --destdir $DATA_BIN \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 20


echo "Start binarizing $TRAIN_BT/train.${SRC}-${TGT}..."    
$PYTHON /home/v-jiaya/unilm-moe/unilm-moe/fairseq/fairseq_cli/preprocess.py  \
    --trainpref $TRAIN_BT/train.${SRC}-${TGT} --validpref $TEST/valid.${SRC}-${TGT} \
    --source-lang $SRC --target-lang $TGT \
    --destdir $DATA_BIN_BT \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 20
