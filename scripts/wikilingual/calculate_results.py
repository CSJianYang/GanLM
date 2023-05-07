import argparse
import os
import io
import sys
import random
import xlwt

LANGS = "en_XX es_XX pt_XX fr_XX de_DE ru_RU it_IT id_ID nl_XX ar_AR vi_VN zh_CN th_TH ja_XX ko_KR hi_IN cs_CZ tr_TR".split()
SIMPLE_LANGS = "en es pt fr de ru it id nl ar vi zh th ja ko hi cs tr".split()


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
    for i in range(len(LANGS)):
        worksheet.write(0, i + 1, label=LANGS[i][:2])
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
                        default=r'/mnt/input/mBART/WikiLingual/download-split/evaluation/baseline/',
                        help='input stream')
    parser.add_argument('--clear-log', '-clear-log', action="store_true", help='input stream')
    parser.add_argument('--excel-path', '-excel-path', type=str, default="/home/v-jiaya/mBART/wikilingual.xls",
                        help='input stream')
    args = parser.parse_args()
    return args


def read_log(model_dir, checkpoint, type="multilinugal"):
    BLEU_results = []
    ROUGE1_results = []
    ROUGE2_results = []
    ROUGEL_results = []
    for lg in SIMPLE_LANGS:
        model_name = "{}/{}/{}".format(model_dir, lg, checkpoint) if type == "monolingual" else "{}/{}".format(
            model_dir, checkpoint)
        file_name = os.path.join(args.log, "BLEU", "{}.BLEU".format(lg))
        clear_log(file_name)
        find_BLEU = False
        with open(file_name, 'r', encoding="utf-8") as r:
            lines = r.readlines()
            for i in range(len(lines) - 1, -1, -1):
                if format_model(model_name) in format_model(lines[i]):
                    BLEU_results.append(round(float(lines[i + 1].split()[2]), 1))
                    find_BLEU = True
                    break
        if not find_BLEU:
            print("BLEU {}: {}".format(checkpoint, lg))
        find_ROUGE = False
        file_name = os.path.join(args.log, "ROUGE", "{}.ROUGE".format(lg))
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
            print("ROUGE {}: {}".format(checkpoint, lg))
        # print("{} | Successfully loading ROUGE and BLEU results...".format(lg))
    assert len(LANGS) == len(BLEU_results) and len(LANGS) == len(ROUGE1_results) and len(LANGS) == len(
        ROUGE2_results) and len(LANGS) == len(ROUGEL_results), "{} | {} | {} | {}".format(len(BLEU_results),
                                                                                          len(ROUGE1_results),
                                                                                          len(ROUGE2_results),
                                                                                          len(ROUGEL_results))
    print(" &".join([str(BLEU_result) for BLEU_result in BLEU_results]))
    print(
        "Avg: {}/{}/{}/{} | {}".format(avg(BLEU_results), avg(ROUGE1_results), avg(ROUGE2_results), avg(ROUGEL_results),
                                       model_dir))
    return BLEU_results, ROUGE1_results, ROUGE2_results, ROUGEL_results


if __name__ == "__main__":
    args = parse_args()
    monolingual_transformer_results = read_log(
        model_dir="/mnt/input/mBART/WikiLingual/download-split/model/monolingual/transformer/",
        checkpoint="avg11_15.pt", type="monolingual")
    monolingual_mBART_results = read_log(
        model_dir="/mnt/input/mBART/WikiLingual/download-split/model/monolingual/mBART/", checkpoint="avg11_15.pt",
        type="monolingual")
    multilingual_transformer_results = read_log(
        model_dir="/mnt/input/mBART/WikiLingual/download-split/model/multilingual/128GPU-LR1e-4/transformer/",
        checkpoint="avg4_8.pt")
    multilingual_mBART_results = read_log(
        model_dir="/mnt/input/mBART/WikiLingual/download-split/model/multilingual/128GPU-LR1e-4/mBART/",
        checkpoint="avg4_8.pt")
    ######Save to Excel###############################
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("wikilingual", cell_overwrite_ok=True)
    create_excel(worksheet, monolingual_transformer_results[0], monolingual_transformer_results[1],
                 monolingual_transformer_results[2], monolingual_transformer_results[3], start_line=1,
                 name="monolingual_transformer")
    create_excel(worksheet, monolingual_mBART_results[0], monolingual_mBART_results[1], monolingual_mBART_results[2],
                 monolingual_mBART_results[3], start_line=3, name="monolingual_mBART")
    create_excel(worksheet, multilingual_transformer_results[0], multilingual_transformer_results[1],
                 multilingual_transformer_results[2], multilingual_transformer_results[3], start_line=5,
                 name="multilingual_transformer")
    create_excel(worksheet, multilingual_mBART_results[0], multilingual_mBART_results[1], multilingual_mBART_results[2],
                 multilingual_mBART_results[3], start_line=7, name="multilingual_mBART")
    workbook.save(args.excel_path)
    print("Scussfully saving results to {}...".format(args.excel_path))
    ######Create Latex File###########################
    latex = r"""
\begin{table*}[t]
\centering
\resizebox{1.0\textwidth}{!}{
\begin{tabular}{l|cccccc}
\toprule
 & en & es & pt & fr & de & ru \\ \midrule
Monolingual Transformer \cite{transformer}  & {} \\ \cmidrule{1-1}
Monolingual mBART \cite{mbart} & {} \\ \cmidrule{1-1}
Multilingual Transformer \cite{transformer}  & {} \\ \cmidrule{1-1}
Multilingual mBART \cite{mbart} & {} \\ \cmidrule{1-1}
\ourmethod{} (Our method) &  {} \\  \midrule
 & it & id & nl & ar & zh & vi \\ \midrule
Transformer \cite{transformer} & {} \\ \cmidrule{1-1}
mBART \cite{mbart} & {} \\ \cmidrule{1-1}
\ourmethod{} (Our method) &  {}  \\ \midrule
 {} \\\midrule
Transformer \cite{transformer} & {} \\ \cmidrule{1-1}\cmidrule{1-1}
mBART \cite{mbart} & {} \\ \cmidrule{1-1}
\ourmethod{} (Our method)  &  {}  \\ \bottomrule
\end{tabular}
}
\caption{Evaluation Results (ROUGE-1/ROUGE-2/ROUGE-L F1
scores) of 18 language on the Wikilingual dataset.}
\label{table:wiki}
\end{table*}"""
