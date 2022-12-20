from fairseq import checkpoint_utils
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from unilm.models.electra_encoder_decoder_v6 import ElectraEncoderDecoderv6
import tqdm
import argparse
from fairseq.data import data_utils, FairseqDataset, Dictionary
TASKS = ["CoLA", "SST-2", "MRPC", "QQP", "STS-B", "MNLI", "QNLI", "RTE", "WNLI", "AX"]
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', '-ckpt-path', type=str,
                        default="/home/v-jiaya/unilm-moe/data/glue/model/CoLA/checkpoint_best.pt",
                        help='main languages file')
    parser.add_argument('--vocab-path', '-vocab-path', type=str,
                        default="/mnt/output/glue/data-bin/CoLA-bin/input0/dict.txt",
                        help='main languages file')
    parser.add_argument('--output', '-output', type=str,
                        default="/home/v-jiaya/unilm-moe/data/glue/model/CoLA/CoLA.tsv",
                        help='main languages file')
    parser.add_argument('--test-set', '-test-set', type=str,
                        default="/home/v-jiaya/unilm-moe/data/glue/glue_data/CoLA/processed/dev",
                        help='main languages file')
    parser.add_argument('--task', '-task', type=str,
                        default="CoLA",
                        help='main languages file')
    parser.add_argument('--label-map', '-label-map', type=str,
                        default="/mnt/output/glue/data-bin/CoLA-bin/label/dict.txt",
                        help='main languages file')
    parser.add_argument('--cuda', '-cuda', action="store_true",
                        help='main languages file')
    args = parser.parse_args()
    print(args)
    return args


def calculate_accuacy(model):
    model.eval()
    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    if os.path.exists(args.label_map):
        label_map = {index: l.strip().split()[0] for index, l in enumerate(open(args.label_map, "r", encoding="utf-8"))}
    else:
        label_map = None
    ncorrect, nsamples = 0, 0
    prediction_labels = []
    if os.path.exists(f"{args.test_set}.label"):
        labels = [l.strip() for l in open(f"{args.test_set}.label", "r", encoding="utf-8")]
    else:
        labels = None

    if os.path.exists(f"{args.test_set}.input1"):
        sents2 = [l.strip() for l in open(f"{args.test_set}.input1", "r", encoding="utf-8")]
    else:
        #sent2 = None
        sents2 = [l.strip() for l in open(f"{args.test_set}.input0", "r", encoding="utf-8")] # sent1 == sent2
        print("Duplicate sent1 and sent2...")
    with open(f"{args.test_set}.input0") as input0_f:
        for index, sent1 in enumerate(input0_f):
            if index % 100 == 0:
                print(f"Processing {index} samples...")
            sent1 = sent1.strip()
            if sents2 is not None:
                sent2 = sents2[index]
                tokens, prev_tokens = model.encode_tokenized(vocab, sent1, sent2)
            else:
                tokens, prev_tokens = model.encode_tokenized(vocab, sent1)
            if args.cuda:
                tokens = tokens.to(device)
                prev_tokens = prev_tokens.to(device)
            if label_map is not None:
                prediction = model.predict('sentence_classification_head', tokens=tokens, prev_tokens=prev_tokens).argmax().item()
                prediction_label = label_map[prediction]
            else:
                prediction_label = model.predict('sentence_classification_head', tokens=tokens, prev_tokens=prev_tokens, return_logits=True)
            prediction_labels.append(prediction_label)
            if labels is not None:
                ncorrect += int(prediction_label == labels[index])
                nsamples += 1
    if labels is not None:
        print(f'{args.task} | Accuracy: {float(ncorrect)/float(nsamples)}')
    else:
        print("No Ground-truth!")
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    with open(args.output, "w", encoding="utf-8") as w:
        for index, prediction_label in enumerate(prediction_labels):
            w.write(f"{index}\t{prediction_label}\n")
    print(f"Saving to {args.output}")


if __name__ == "__main__":
    args=parse_args()
    assert args.task in TASKS
    vocab = Dictionary.load(args.vocab_path)
    vocab.add_symbol("<mask>")
    for i in range(100):
        vocab.add_symbol(f"<mask_{i}>")
    #vocab.pad_to_multiple_(padding_factor=8)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt_path)
    model = ElectraEncoderDecoderv6.build_model(state["cfg"].model, src_dict=vocab, tgt_dict=vocab)
    num_classes = 3 if args.task != "STS-B" else 1
    model.register_classification_head(
        "sentence_classification_head",
        num_classes=num_classes,
    )
    model.load_state_dict(state["model"], strict=True)
    print(f"Successfully loading model from {args.ckpt_path}")
    calculate_accuacy(model)
