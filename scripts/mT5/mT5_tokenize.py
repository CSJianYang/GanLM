import sys
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("/home/v-jiaya/unilm-moe/data/PretrainedModels/mT5/mt5-base/")
for line in sys.stdin:
    sys.stdout.write(tokenizer.tokenize(line))
