LANGS=(en fr es de el bg ru tr ar vi th zh hi sw ur)
DICT=/mnt/output/XNLI-XLMR/data-bin/en/input0/dict.txt
for lg in ${LANGS[@]}; do
    echo "${DICT} - > ${lg}"
    cp $DICT /mnt/output/XNLI-XLMR/data-bin/${lg}/input0/
    cp $DICT /mnt/output/XNLI-XLMR/data-bin/${lg}/input1/
done