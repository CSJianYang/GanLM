# import os
from sys import argv
import json


lang = argv[1]
lines = open(argv[2]).read().strip().split('\n')

templates = json.load(open(argv[3]))

with open('test_data/outs.json', 'w') as fo, open('test_data/zcode-refs.json', 'w') as fr:
    output = {}
    refs = {}
    output['language'] = lang
    refs['language'] = lang
    output['task'] = 'webnlg'
    refs['task'] = 'webnlg'
    output['values'] = []
    refs['values'] = []
    assert len(lines) == len(templates['values'])
    for sample, temp in zip(lines, templates['values']):
        for i in range(len(temp['target'])):
            output['values'].append({"generated": sample})
            refs['values'].append({"target": temp['target'][i]})
    print(json.dumps(output, indent=4, ensure_ascii=False), file=fo)
    print(json.dumps(refs, indent=4, ensure_ascii=False), file=fr)
