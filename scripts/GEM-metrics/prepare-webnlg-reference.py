from datasets import load_dataset
import os
from sys import argv
import json

lang = argv[1]
webnlg = load_dataset('gem', f'web_nlg_{lang}')

for split in ['validation', 'test']:
    with open(f'test_data/webnlg_{lang}_{split}.json', 'w') as f:
        output = {}
        output['language'] = lang
        output['values'] = []
        for sample in webnlg[split]:
            output['values'].append({"target": sample['references']})
        print(json.dumps(output, indent=4, ensure_ascii=False), file=f)
        
        