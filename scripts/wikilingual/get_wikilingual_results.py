import argparse
import os

SIMPLE_LANGS = "ar cs de en es fr hi it id ja ko nl pt ru th tr vi zh".split()


def avg(results):
    return round(sum(results) / len(results), 1)


def format_model(model, **extra_args):
    if len(extra_args) > 0:
        return "{} | beam: {} | min_len: {} | max_len: {} | len_pen: {}".format(model, extra_args['beam'],
                                                                                extra_args['min_len'],
                                                                                extra_args['max_len'],
                                                                                extra_args['len_pen'])
    else:
        return model.replace("//", "/")


def create_excel(worksheet, BLEU_results, ROUGE1_results, ROUGE2_results, ROUGEL_results, start_line=3, name="mBART"):
    worksheet.write(start_line, 0, label="BLEU ({})".format(name))
    worksheet.write(start_line + 1, 0, label="ROUGE1/ROUGE2/ROUGEL ({})".format(name))
    for i in range(len(SIMPLE_LANGS)):
        worksheet.write(0, i + 1, label=SIMPLE_LANGS[i][:2])
        worksheet.write(start_line, i + 1, label=BLEU_results[i])
        worksheet.write(start_line + 1, i + 1,
                        label="{}/{}/{}".format(ROUGE1_results[i], ROUGE2_results[i], ROUGEL_results[i]))
    worksheet.write(0, len(BLEU_results) + 1, "avg")
    worksheet.write(start_line, len(BLEU_results) + 1, label=round(avg(BLEU_results), 2))
    worksheet.write(start_line + 1, len(BLEU_results) + 1,
                    label="{}/{}/{}".format(round(avg(ROUGE1_results), 1), round(avg(ROUGE2_results), 1),
                                            round(avg(ROUGEL_results), 1)))


def clear_log(file_name):
    with open(file_name, "r", encoding="utf-8") as r:
        lines = r.readlines()
        new_lines = []
        for i in range(len(lines)):
            if (".pt | beam" in lines[i] and i == len(lines) - 1) or (
                    ".pt | beam" in lines[i] and ".pt | beam" in lines[i + 1]):
                continue
            else:
                new_lines.append(lines[i].replace("//", "/"))
    if len(lines) != len(new_lines):
        print("Clearing {}: {} -> {}".format(file_name, len(lines), len(new_lines)))
        with open(file_name, "w", encoding="utf-8") as w:
            w.write("".join(new_lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str,
                        default=r'/mnt/output/wikilingual/ROUGE/mbart/', help='input stream')
    parser.add_argument('--clear-log', '-clear-log', action="store_true", help='input stream')
    parser.add_argument('--excel-path', '-excel-path', type=str, default="/home/v-jiaya/mBART/wikilingual.xls",
                        help='input stream')
    args = parser.parse_args()
    return args


def read_log(model_name):
    BLEU_results = []
    ROUGE1_results = []
    ROUGE2_results = []
    ROUGEL_results = []
    for lg in SIMPLE_LANGS:
        find_ROUGE = False
        file_name = os.path.join(args.log, f"ROUGE.{lg}")
        clear_log(file_name)
        with open(file_name, 'r', encoding="utf-8") as r:
            lines = r.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if format_model(model_name) in format_model(lines[i]):
                    break
            results_lines = lines[i + 3: i + 15]
            for line in results_lines:
                if "ROUGE-1 Average_F:" in line:
                    ROUGE1_results.append(round(float(line.split()[5]) * 100, 1))
                elif "ROUGE-2 Average_F:" in line:
                    ROUGE2_results.append(round(float(line.split()[5]) * 100, 1))
                elif "ROUGE-L Average_F:" in line:
                    ROUGEL_results.append(round(float(line.split()[5]) * 100, 1))
                    find_ROUGE = True
                    break
        if not find_ROUGE:
            print("ROUGE {}: {}".format(model_name, lg))
        # print("{} | Successfully loading ROUGE and BLEU results...".format(lg))
    assert len(SIMPLE_LANGS) == len(ROUGE1_results) and len(SIMPLE_LANGS) == len(ROUGE2_results) and len(
        SIMPLE_LANGS) == len(ROUGEL_results), "{} | {} | {} | {}".format(len(BLEU_results), len(ROUGE1_results),
                                                                         len(ROUGE2_results), len(ROUGEL_results))
    # print(" &".join([str(BLEU_result) for BLEU_result in BLEU_results]))
    print(f"Avg: {avg(ROUGE1_results)}/{avg(ROUGE2_results)}/{avg(ROUGEL_results)}")
    return ROUGE1_results, ROUGE2_results, ROUGEL_results


if __name__ == "__main__":
    args = parse_args()
    multilingual_transformer_results = read_log(model_name="mbart")
    ######Save to Excel###############################
    # workbook = xlwt.Workbook(encoding='utf-8')
    # worksheet = workbook.add_sheet("wikilingual", cell_overwrite_ok=True)
    # create_excel(worksheet, multilingual_transformer_results[0], multilingual_transformer_results[1], monolingual_transformer_results[2], monolingual_transformer_results[3], start_line=1, name="monolingual_transformer")
    # workbook.save(args.excel_path)
    # print("Scussfully saving results to {}...".format(args.excel_path))
    ######Create Latex File###########################
    latex = []
    for i, lg in enumerate(SIMPLE_LANGS):
        latex.append(
            f"{multilingual_transformer_results[0][i]}/{multilingual_transformer_results[1][i]}/{multilingual_transformer_results[2][i]}")
    latex = " & ".join(latex)
    print(latex)
