from transformers import MT5Model, T5Tokenizer
import torch.nn as nn
import torch
from transformers.tokenization_utils_base import BatchEncoding
model = MT5Model.from_pretrained("/home/v-jiaya/unilm-moe/data/PretrainedModels/mT5/mt5-base/")
tokenizer = T5Tokenizer.from_pretrained("/home/v-jiaya/unilm-moe/data/PretrainedModels/mT5/mt5-base/")
article = ["UN Offizier sagt, dass "]
summary = ["Weiter Verhandlung"]
inputs = tokenizer(article, return_tensors="pt", padding=True)
with tokenizer.as_target_tokenizer():
    labels = tokenizer(summary, return_tensors="pt", padding=True)
inputs.data['input_ids'] = inputs.data['input_ids'][:,:-1]
inputs.data['attention_mask'] = inputs.data['attention_mask'][:,:-1]
labels.data['input_ids'] = labels.data['input_ids'][:,:-1]
labels.data['attention_mask'] = labels.data['attention_mask'] [:,:-1]
model.eval()
encoder_outputs = model.encoder(input_ids=inputs["input_ids"])
outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
#tensor([[[-1.4654e-01,  1.0535e-01, -3.1200e-01,  ..., -1.4037e-01,
         #  -2.5364e-01,  3.8881e-01],
         # [-6.0841e-02, -2.1513e-01,  1.5812e-01,  ..., -2.7600e-02,
         #  -2.1247e-01,  7.4309e-02],
         # [ 3.0147e-01, -3.2947e-01,  2.6326e-02,  ...,  1.9128e-01,
         #   3.0275e-01,  3.2382e-02],
         # ...,
         # [ 7.3470e-02,  2.2633e-01, -3.0625e-01,  ...,  5.3023e-02,
         #  -3.4938e-01,  1.5530e-01],
         # [ 4.8176e-05,  7.5503e-03,  1.3400e-02,  ...,  2.0571e-04,
         #  -2.7096e-03, -3.3077e-03],
         # [-1.1923e-01, -2.3377e-02, -1.9513e-02,  ...,  1.9479e-04,
         #   1.7491e-01,  1.0937e-04]]], grad_fn=<MulBackward0>)
output_projection = nn.Linear(
    model.encoder.embed_tokens.weight.shape[1],
    model.encoder.embed_tokens.weight.shape[0],
    bias=False,
)
#output_projection.weight = model.encoder.embed_tokens.weight
#y = outputs.last_hidden_state * (768 ** -0.5)
y = output_projection(outputs.last_hidden_state)
torch.log_softmax(y, dim=-1)
#inputs = tokenizer(article, return_tensors="pt", padding=True)
#model.encoder.encode(**inputs)
#tensor([   320,    921,   9453,    259,    262,   6329,    304,    259, 142240,
         # 66089,    287, 176374,    304,   6117,    332,    265,  75807,    305,
         #   287,   3004,    263,    304,    772,  31521,    260,  16956,    772,
         #   259,  75807,    259,  25386,    772,   3004,    260,  80998,    772,
         #   259,  75807,    281,    259,    262,  25720,  74187,    259,  76114,
         #   286,  52142,    260,      2])
#tokenizer.sp_model.piece_to_id(["'▁S", 'que', 'eze', '▁', 'a', '▁line', '▁of', '▁', 'lotion', '▁onto', '▁the', '▁tops', '▁of', '▁both', '▁for', 'e', 'arms', '▁and', '▁the', '▁back', 's', '▁of', '▁your', '▁hands', '.', '▁Place', '▁your', '▁', 'arms', '▁', 'behind', '▁your', '▁back', '.', '▁Move', '▁your', '▁', 'arms', '▁in', '▁', 'a', '▁wind', 'shield', '▁', 'wipe', 'r', '▁motion', ".'"])


# from transformers import TFMT5ForConditionalGeneration, T5Tokenizer
#
# model = TFMT5ForConditionalGeneration.from_pretrained("/home/v-jiaya/unilm-moe/data/PretrainedModels/mT5/mt5-base/")
# tokenizer = T5Tokenizer.from_pretrained("/home/v-jiaya/unilm-moe/data/PretrainedModels/mT5/mt5-base/")
# article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
# summary = "Weiter Verhandlung in Syrien."
# inputs = tokenizer(article, return_tensors="tf")
# with tokenizer.as_target_tokenizer():
#     labels = tokenizer(summary, return_tensors="tf")
#
# outputs = model(**inputs, labels=labels["input_ids"])