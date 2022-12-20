import argparse
import xlwt
import xlrd
import os
from collections import OrderedDict

TOTAL_DIRECTION = 30
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
LANGS="de en it nl ro".split()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str,
                        default=r'/mnt/output/iwslt17/BLEU/mbart/', help='input stream')
    parser.add_argument('--checkpoint-name', '-checkpoint-name', type=str,
                        default=r'mbart', help='input stream')
    args = parser.parse_args()
    return args


def create_excel(results, name, save_dir = '/home/v-jiaya/SharedTask/SmallTask1_ExcelResults/'):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(name, cell_overwrite_ok=True)
    worksheet.write(1, 0, label="DeltaLM-Postnorm (Large)")
    for i in range(len(LANGS)):
        worksheet.write(0, i + 1, label=LANGS[i])
        worksheet.write(i + 1, 0, label=LANGS[i])
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            worksheet.write(i + 1, j + 1, label=results[i][j])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    workbook.save('{}/{}.xls'.format(save_dir, name))
    return workbook


def _lang_pair(src, tgt):
    return "{}->{}".format(src, tgt)


def calculate_avg_score(x2x, src=None, tgt=None, model_name="m2m"):
    results = []
    if src == "x" and tgt == "y":
        name_list = ["de-it", "it-de", "nl-ro", "ro-nl", "de-nl", "nl-de"]
        for name in name_list:
            results.append(x2x[name])
        x2x_results = []
        for key in x2x.keys():
            if "en" not in key:
                x2x_results.append(x2x[key])
        avg = sum(x2x_results) / len(x2x_results)
        results.append(round(avg, 1))
        print("{}: x-y: {:.2f}".format(model_name, avg))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    elif src == "en" and tgt == "en":
        name_list = ["en-de", "de-en", "en-it", "it-en", "en-nl", "nl-en", "en-ro", "ro-en"]
        for name in name_list:
            results.append(x2x[name])
        avg = sum(results) / len(results)
        print(f"{model_name}: en-x/x-en: {avg}")
        results.append(round(avg, 1))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    elif src is not None:
        for key in x2x.keys():
            if "{}->".format(src) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)
    else:
        for key in x2x.keys():
            results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: all: {:.2f}".format(model_name, avg))
        results.append(round(avg, 2))
        results = [str(result) for result in results]
        output = " & ".join(results) + " \\\\"
        print(output)



if __name__ == "__main__":
    args = parse_args()
    x2x = {}
    results = []
    checkpoint_name = args.checkpoint_name
    print("MODEL: {}".format(checkpoint_name))
    for src in LANGS:
        for tgt in LANGS:
            if src == tgt:
                continue
            with open(f"{args.log}/BLEU.{src}-{tgt}", "r", encoding="utf-8") as r:
                result_lines = r.readlines()
                for i in range(len(result_lines) - 1, -1, -1):  # reversed search
                    if checkpoint_name.replace("//", "/") in result_lines[i].strip().replace("//", "/").replace("MODEL: ", ""):
                        last_line = result_lines[i + 1]  # read the latest results
                        if 'BLEU+case.mixed' in last_line:
                            score = float(last_line.split()[2])
                            x2x["{}-{}".format(src, tgt)] = score
                            results.append(score)
                            break
                        else:
                            print(os.path.join(args.log, "{}-{}.BLEU".format(src, "en")))
                            break
    calculate_avg_score(x2x, src="en", tgt="en", model_name="our")
    calculate_avg_score(x2x, src="x", tgt="y", model_name="our")

    #name = "wmt10"
    #create_excel(results, name=name)







