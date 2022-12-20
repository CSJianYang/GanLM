import linecache
DATA="/home/v-jiaya/unilm-moe/data/wikilingual/download-split/train_spm/"
LANGS="ar cs de en es fr hi it id ja ko nl pt ru th tr vi zh".split()
for lg in LANGS:
    src_lines = linecache.getlines(f"{DATA}/train.{lg}_src")
    tgt_lines = linecache.getlines(f"{DATA}/train.{lg}_tgt")
    src_tokens = []
    tgt_tokens = []
    for i in range(len(src_lines)):
        src_len = len(src_lines[i].split())
        tgt_len = len(tgt_lines[i].split())
        src_tokens.append(src_len)
        tgt_tokens.append(tgt_len)

    print(f"{lg}-SRC: Avg: {sum(src_tokens) // len(src_tokens)} | Max: {max(src_tokens)}")
    print(f"{lg}-TGT: Avg: {sum(tgt_tokens) // len(tgt_tokens)} | Max: {max(tgt_tokens)}")
