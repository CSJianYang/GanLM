export PYTHONWARNINGS="ignore"

OUTPUT_PATH=${1}
CHECKPOINT_NAME=${2}
bsz=${3}
beam=${4}
lenpen=${5}
src=${6}
BLEU_DIR=${7}
EVAL_SCRIPT=./shells/mt/flores/evaluate_flores.sh


#TGTS=(af)
TGTS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
for tgt in ${TGTS[@]}; do
    if [ $src != $tgt ]; then
        echo "${src}->${tgt}"
        bash $EVAL_SCRIPT  ${OUTPUT_PATH} ${CHECKPOINT_NAME} ${bsz} ${beam} ${lenpen} ${src} ${tgt} ${BLEU_DIR}
    fi
done
