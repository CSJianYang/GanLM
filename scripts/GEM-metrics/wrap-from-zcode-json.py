import os
from sys import argv
import json

lang = argv[1]
lines = json.load(open(argv[2]))
lines = [l['prediction'][0] for l in lines if l['cid'].endswith(lang)]

with open('test_data/outs.json', 'w') as f:
    output = {}
    output['language'] = lang
    output['task'] = 'webnlg'
    output['values'] = []
    for sample in lines:
        output['values'].append({"generated": sample})
    print(json.dumps(output, indent=4, ensure_ascii=False), file=f)
        
        