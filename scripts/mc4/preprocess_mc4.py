import argparse
import os
import gzip
import shutil
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-dir', type=str,
                        default=r'/mnt/output/Data/multilingual/', help='input stream')
    parser.add_argument('--decompress-dir', '-decompress-dir', default="/mnt/output/Data/mc4/decompress_mc4/",
                        help='input stream')
    parser.add_argument('--new-dir', '-new-dir', default="/mnt/output/Data/mc4/raw_mc4/", help='input stream')
    parser.add_argument('--lg', '-lg', default="en", help='input stream')
    parser.add_argument('--index', '-index', default="0-500", help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    file_names = os.listdir(args.dir)
    start, end = args.index.split("-")
    start = int(start)
    end = int(end)
    if args.lg != "all":
        file_names = list(filter(lambda x: args.lg in x, file_names))
        file_names = list(filter(lambda x: start <= int(x.split(".")[1].split("-")[1]) <= end, file_names))
    print(f"Total {len(file_names)} files")
    for file_id, file_name in enumerate(file_names):
        lg = file_name.split(".")[0].split("-")[1]
        if not os.path.exists(f"{args.decompress_dir}/{lg}"):
            os.makedirs(f"{args.decompress_dir}/{lg}")
        if not os.path.exists(f"{args.new_dir}/{lg}"):
            os.makedirs(f"{args.new_dir}/{lg}")
        file = f"{args.dir}/{file_name}"
        decompress_file = f"{args.decompress_dir}/{lg}/{file_name}".replace(".gz", "")
        new_file = f"{args.new_dir}/{lg}/{file_name}".replace(".gz", ".txt")
        with gzip.open(file, 'rb') as f_in:
            with open(decompress_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        with open(decompress_file, "r", encoding="utf-8") as f_in:
            with open(new_file, 'w') as f_out:
                for line in f_in:
                    json_text = json.loads(line)
                    f_out.write(f"{json_text['text']}\n\n")
        print(f"{file_id} | Successfully saving to {new_file}")
