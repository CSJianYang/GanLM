import os
import glob
import json

root = "/mnt/output/xsum/model/electra-encoder-decoder-v5/lr5e-4-bsz8192-ws10000-wd0.01-dw10_1000_-1-iw1.0_1000_-1" \
       "-g12d6--share-generator-discriminator/checkpoint_1_175000-ft/"

dirs = os.listdir(root)
rouge_scores = []
for dir in dirs:
    rouge_dir = f"{root}/{dir}/generator/"
    rouge_files = glob.glob(f"{rouge_dir}/*.rouge")
    for rouge_file in rouge_files:
        rouge_scores.append((rouge_file, json.load(open(rouge_file))))
rouge_file = ""
rouge1 = 0
rouge2 = 0
for rouge_score in rouge_scores:
    if rouge_score[1]["rg1"] + rouge_score[1]["rg2"] > rouge1 + rouge2:
        rouge_file = rouge_score[0]
        rouge1 = rouge_score[1]["rg1"]
        rouge2 = rouge_score[1]["rg2"]
print(rouge_file)
print(f"{rouge1}/{rouge2}")
