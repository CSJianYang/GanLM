import os


LANGS = "en es pt fr de ru it id nl ar vi zh th ja ko hi cs tr".split()
mBART_LANGS = "en_XX es_XX pt_XX fr_XX de_DE ru_RU it_IT id_ID nl_XX ar_AR vi_VN zh_CN th_TH ja_XX ko_KR hi_IN cs_CZ tr_TR".split()
mBART_OUTPUT_DIR = "/home/v-jiaya/mBART/mBART/shells/aml/multi-node/WikiLingual/8GPU/monolingual/mBART/"
mono_OUTPUT_DIR = "/home/v-jiaya/mBART/mBART/shells/aml/multi-node/WikiLingual/8GPU/monolingual/transformer/"
for i in range(len(LANGS)):
    cmds = """TEXT=/mnt/input/mBART/WikiLingual/download-split/data-bin-mBART/
MODEL=/mnt/input/mBART/WikiLingual/download-split/model/monolingual/mBART/%s/
PRETRAINED_MODEL=/mnt/input/mBART/PretrainedModel/mbart50.pretrained/model.pt
mkdir -p $MODEL
if [ ! -f $MODEL/checkpoint_last.pt ]; then #To Fit the Preemptible Job
    echo "Start Training From the Pretrained Model (Reset the Optimizer, Meters, Lr-Schedule, and Dataloader)..."
    python train.py $TEXT \\
        --save-dir $MODEL --restore-file $PRETRAINED_MODEL --arch mbart_large --encoder-normalize-before --decoder-normalize-before  \\
        --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \\
        --encoder-langtok "src" --decoder-langtok --langtoks '{"main":("src","tgt")}' \\
        --langs "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI" \\
        --lang-pairs "%s-%s" --truncate-source --dropout 0.1 --attention-dropout 0.1 \\
        --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\
        --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 5e-5 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \\
        --max-update 400000 --max-epoch 15 --weight-decay 0.0 --max-tokens 1536 --update-freq 1 \\
        --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 \\
        --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log
else
    echo "Start Training From the Previous Checkpoint (Do not Reset the Optimizer, Meters, Lr-Schedule, and Dataloader)..."
    python train.py $TEXT \\
        --save-dir $MODEL --arch mbart_large --encoder-normalize-before --decoder-normalize-before  \\
        --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \\
        --encoder-langtok "src" --decoder-langtok --langtoks '{"main":("src","tgt")}' \\
        --langs "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI" \\
        --lang-pairs "%s-%s" --truncate-source --dropout 0.1 --attention-dropout 0.1 \\
        --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\
        --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 5e-5 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \\
        --max-update 400000 --max-epoch 15 --weight-decay 0.0 --max-tokens 1536 --update-freq 1 \\
        --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log
fi""" % (LANGS[i], mBART_LANGS[i], mBART_LANGS[i], mBART_LANGS[i], mBART_LANGS[i])
    with open(os.path.join(mBART_OUTPUT_DIR, "train_mBART_{}.sh".format(LANGS[i])), "w") as w:
        w.write(cmds)

    cmds = """TEXT=/mnt/input/mBART/WikiLingual/download-split/data-bin-mBART/
MODEL=/mnt/input/mBART/WikiLingual/download-split/model/monolingual/mono/%s/
echo "Start Training ..."
python train.py $TEXT \\
    --save-dir $MODEL --arch mbart_large --encoder-normalize-before --decoder-normalize-before  \\
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \\
    --encoder-langtok "src" --decoder-langtok --langtoks '{"main":("src","tgt")}' \\
    --langs "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI" \\
    --lang-pairs "%s-%s" --truncate-source --dropout 0.1 --attention-dropout 0.1 \\
    --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \\
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 5e-5 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \\
    --max-update 400000 --max-epoch 15 --weight-decay 0.0 --max-tokens 1536 --update-freq 1 \\
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log""" % (
    LANGS[i], mBART_LANGS[i], mBART_LANGS[i])
    with open(os.path.join(mono_OUTPUT_DIR, "train_mono_{}.sh".format(LANGS[i])), "w") as w:
        w.write(cmds)

heads = """#docker's key: /r9XLj/sS40pDvIvzamSeWZHApMhEc1r
description: mBART

target:
    service: amlk8s
    #name: itplabrr1cl1
    #service: amlk8s
    #vc: resrchvc
    name: itpwus2v100cl
    vc: gcrprojvc1
    #vc: resrchvc  
    #name: v100-8x-eus-1
    #vc: quantus

environment:
    image: nvidia/20.09:v7.0.2
    registry: shumingdocker.azurecr.io
    setup:
    - ibstat
    - ulimit -n 4096
    - python -m pip install --editable . --user
    username: shumingdocker
  
storage:
    output:
        storage_account_name: yangjianblob
        container_name: phillytools
    input:
        storage_account_name: yangjianblob
        container_name: phillytools

code:
    # local directory of the code. this will be uploaded to the server.
    # $CONFIG_DIR is expanded to the directory of this config file
    local_dir: $CONFIG_DIR/
jobs:"""
cmds = heads
for i in range(len(LANGS)):
    cmds += """
- name: wikilingual_mono_%s
  sku: G8
  sku_count: 1
  command:
     - bash ./shells/aml/multi-node/WikiLingual/8GPU/monolingual/mono/train_mono_%s.sh""" % (LANGS[i], LANGS[i])
mono_train_scripts = "/home/v-jiaya/mBART/mBART/submit_train_wikilingual_monolingual_mono.yaml"
with open(mono_train_scripts, "w", encoding="utf-8") as w:
    w.write(cmds)

cmds = heads
for i in range(len(LANGS)):
    cmds += """
- name: wikilingual_mBART_%s
  sku: G8
  sku_count: 1
  command:
    - bash ./shells/aml/multi-node/WikiLingual/8GPU/monolingual/mBART/train_mBART_%s.sh""" % (LANGS[i], LANGS[i])
mBART_train_scripts = "/home/v-jiaya/mBART/mBART/submit_train_wikilingual_monolingual_mBART.yaml"
with open(mBART_train_scripts, "w", encoding="utf-8") as w:
    w.write(cmds)
