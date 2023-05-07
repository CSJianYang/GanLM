# import os
from sys import argv
import json


lang = argv[1]
lines = open(argv[2]).read().strip().split('\n')
output_file=argv[3]

with open(output_file, 'w') as f:
    output = {}
    output['language'] = lang
    output['task'] = 'webnlg'
    output['values'] = []
    for sample in lines:
        output['values'].append({"generated": sample})
    print(json.dumps(output, indent=4, ensure_ascii=False), file=f)
