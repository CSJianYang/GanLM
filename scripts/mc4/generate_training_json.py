import argparse
import os
import gzip
import shutil
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', '-input-json', type=str,
                        default=r'/mnt/output/Data/deltalm/json/train.json', help='input stream')
    parser.add_argument('--output-json', '-output-json', default="mnt/output/Data/mc4_pretrain/json/train.json",
                        help='input stream')
    parser.add_argument('--data', '-data', default="/mnt/output/Data/mc4/raw_mc4/", help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    json_text = json.load(open(args.input_json))
    lg_index = {}
    for i in range(len(json_text)):
        lg_index[json_text[i]["source_lang"]] = i
    lgs = os.listdir(args.data)
    for lg in lgs:
        file_names = os.listdir(f"{args.data}/{lg}/")
        file_names = list(filter(lambda x: "validation" not in x, file_names))
        files = []
        for file_name in file_names:
            files.append(f"raw_mc4/{lg}/{file_name}")
        if lg in lg_index:
            json_text[lg_index[lg]]["source"] += files
            print(f"Adding {lg} into json file")
        else:
            print(f"Can not find {lg} in json file")
        #     json_text.append({
        #         'source': [],
        #         'source_lang': lg,
        #         'type': 'raw',
        #         'name': lg,
        #         'weight': 0,
        #     })
    json.dumps(json_text)
