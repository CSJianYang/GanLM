from datasets import load_dataset
import os
from sys import argv
import json

lang = argv[1]
wikilingua = load_dataset('gem', f'wiki_lingua_{lang}_en')

for split in ['validation', 'test']:
    with open(f'test_data/wiki_lingua_{lang}_{split}.json', 'w') as f:
        output = {}
        output['language'] = lang
        output['values'] = []
        for sample in wikilingua[split]:
            output['values'].append({"target": sample['references']})
        print(json.dumps(output, indent=4, ensure_ascii=False), file=f)
        
        