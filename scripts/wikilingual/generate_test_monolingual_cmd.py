import argparse
import os
import io
import sys
import pickle
import random

LANGS = "en es pt fr de ru it id nl ar vi zh th ja ko hi cs tr".split()
mBART_LANGS = "en_XX es_XX pt_XX fr_XX de_DE ru_RU it_IT id_ID nl_XX ar_AR vi_VN zh_CN th_TH ja_XX ko_KR hi_IN cs_CZ tr_TR".split()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer-checkpoint', '-transformer-checkpoint', type=str, default="/mnt/input/mBART/WikiLingual/download-split/model/monolingual/transformer", help='output stream')
    parser.add_argument('--mBART-checkpoint', '-mBART-checkpoint', type=str, default="/mnt/input/mBART/WikiLingual/download-split/model/monolingual/mBART", help='output stream')
    parser.add_argument('--transformer-output', '-transformer-output', type=str,
                        default=r'/home/v-jiaya/mBART/mBART/submit_test_wikilingual_monolingual_transformer.yaml', help='output stream')
    parser.add_argument('--mBART-output', '-mBART-output', type=str,
                        default=r'/home/v-jiaya/mBART/mBART/submit_test_wikilingual_monolingual_mBART.yaml', help='output stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    heads = """#docker's key: /r9XLj/sS40pDvIvzamSeWZHApMhEc1r
description: test        
target:
  service: amlk8s
  #name: itplabrr1cl1
  #name: itpeastusv100cl
  #vc: resrchvc
  name: v100-32gb-eus-2
  vc: language-itp-mtres
  #name: v100-8x-scus
  #name: v100-8x-eus-1
  #vc: quantus


environment:
  image: nvidia/20.09:v7.0.2
  registry: shumingdocker.azurecr.io
  setup:
  - ibstat
  - ulimit -n 4096
  - python -m pip install --editable . --user
  - python -m pip install -U git+https://github.com/pltrdy/pyrouge --user
  - git clone https://github.com/pltrdy/files2rouge.git && cd files2rouge && echo '/home/t-jianya/.files2rouge/' | python setup_rouge.py && python setup.py install --user
  #- python -m pip install -U nltk --user && python -m nltk.downloader all > nltk.log
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
    beam = 3
    batch_size = 24
    min_len = 3
    max_len = 196
    checkpoint_dir = args.transformer_checkpoint

    cmds  = heads
    for i in range(len(LANGS)):
        checkpoint = "{}/{}/avg11_15.pt".format(checkpoint_dir, LANGS[i])
        checkpoint_name = checkpoint.split("model")[1].replace(".", "_").replace("/", "_")
        cmds += """
- name: test_monolingual_transformer_{}_{}
  sku: G1
  sku_count: 1
  command:
    - bash ./shells/aml/multi-node/WikiLingual/test/test_aml.sh {} {} {} {} {} {} {} {}""".format(checkpoint.replace(".", "_"), LANGS[i], mBART_LANGS[i], LANGS[i], batch_size, beam, checkpoint, checkpoint_name, min_len, max_len)

    print(cmds)
    with open(args.transformer_output, "w", encoding="utf-8") as w:
        w.write(cmds)

    cmds  = heads
    checkpoint_dir = args.mBART_checkpoint
    for i in range(len(LANGS)):
        checkpoint = "{}/{}/avg11_15.pt".format(checkpoint_dir, LANGS[i])
        checkpoint_name = checkpoint.split("model")[1].replace(".", "_").replace("/", "_")
        cmds += """
- name: test_monolingual_mBART_{}_{}
  sku: G1
  sku_count: 1
  command:
    - bash ./shells/aml/multi-node/WikiLingual/test/test_aml.sh {} {} {} {} {} {} {} {}""".format(checkpoint.replace(".", "_"), LANGS[i], mBART_LANGS[i], LANGS[i], batch_size, beam, checkpoint, checkpoint_name, min_len, max_len)

    print(cmds)
    with open(args.mBART_output, "w", encoding="utf-8") as w:
        w.write(cmds)




