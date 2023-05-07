import sys
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
for line in sys.stdin:
    tokenized_line = tokenizer.tokenize(line)
    sys.stdout.write(tokenized_line)
