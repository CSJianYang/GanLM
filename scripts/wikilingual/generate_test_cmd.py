import argparse
LANGS="en es pt fr de ru it id nl ar vi zh th ja ko hi cs tr".split()
mBART_LANGS="en_XX es_XX pt_XX fr_XX de_DE ru_RU it_IT id_ID nl_XX ar_AR vi_VN zh_CN th_TH ja_XX ko_KR hi_IN cs_CZ tr_TR".split()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multilingual-checkpoint', '-multilingual-checkpoint', type=str, 
                        default="/mnt/input/mBART/WikiLingual/download-split/model/multilingual/128GPU-LR1e-4/transformer/avg4_8.pt", help='output stream')
    parser.add_argument('--mBART-checkpoint', '-mBART-checkpoint', type=str,
                        default="/mnt/input/mBART/WikiLingual/download-split/model/multilingual/128GPU-LR1e-4/mBART/avg4_8.pt", help='output stream')
    parser.add_argument('--multilingual-output', '-multilingual-output', type=str,
                        default=r'/home/v-jiaya/mBART/mBART/submit_test_wikilingual_multilingual_transformer.yaml',  help='output stream')
    parser.add_argument('--mBART-output', '-mBART-output', type=str,
                        default=r'/home/v-jiaya/mBART/mBART/submit_test_wikilingual_multilingual_mBART.yaml', help='output stream')
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
  #name: v100-8x-eus-1
  #name: v100-8x-scus
  #vc: quantus
  name: v100-32gb-eus-2  
  vc: language-itp-mtres
  


environment:
  image: nvidia/20.09:v7.0.2
  registry: shumingdocker.azurecr.io
  setup:
  - ibstat
  - ulimit -n 4096
  - python -m pip install --editable . --user
  - python -m pip install -U git+https://github.com/pltrdy/pyrouge --user
  - git clone https://github.com/pltrdy/files2rouge.git && cd files2rouge && echo '/tmp/code/.files2rouge/' | python setup_rouge.py && python setup.py install --user
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
jobs:      
    """
    beam = 3
    batch_size = 32
    min_len = 2
    max_len = 128
    cmds = heads
    checkpoint = args.multilingual_checkpoint
    for i in range(len(LANGS)):
        cmds += """
- name: test_{}_{}
  sku: G1
  sku_count: 1
  command:
    - bash ./shells/aml/multi-node/WikiLingual/test/test_aml.sh {} {} {} {} {} {} {}""".format(checkpoint.replace(".", "_"), LANGS[i], mBART_LANGS[i], LANGS[i], batch_size, beam, checkpoint, min_len, max_len)
    with open(args.multilingual_output, "w", encoding="utf-8") as w:
        w.write(cmds)

    cmds = heads
    checkpoint = args.mBART_checkpoint
    for i in range(len(LANGS)):
        cmds += """
- name: test_{}_{}
  sku: G1
  sku_count: 1
  command:
    - bash ./shells/aml/multi-node/WikiLingual/test/test_aml.sh {} {} {} {} {} {} {}""".format(checkpoint.replace(".", "_"), LANGS[i], mBART_LANGS[i], LANGS[i], batch_size, beam, checkpoint, min_len, max_len)
    with open(args.mBART_output, "w", encoding="utf-8") as w:
        w.write(cmds)




