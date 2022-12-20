from fairseq import checkpoint_utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from unilm.models.electra_encoder_decoder_v6 import ElectraEncoderDecoderv6
import tqdm
import argparse
from fairseq.data import data_utils, FairseqDataset, Dictionary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', '-ckpt-path', type=str,
                        default="/home/v-jiaya/unilm-moe/data/xnli/lr3e-5-uf4-bs16-sd1-ws4000-cn0.0-wd0.1/discriminator/checkpoint_best.pt",
                        help='main languages file')
    parser.add_argument('--vocab-path', '-vocab-path', type=str,
                        default="/mnt/output/XNLI-XLMR/data-bin/en/input0/dict.txt",
                        help='main languages file')
    parser.add_argument('--lgs', '-lgs', type=str,
                        default="/mnt/output/XNLI-XLMR/data-bin/en/input0/dict.txt",
                        help='main languages file')
    args = parser.parse_args()
    print(args)
    return args


def calculate_accuacy(model, filename, lg):
    model.eval()
    model.cuda()
    label_map = {0: 'contradiction', 1: 'entailment', 2: 'neutral', }
    ncorrect, nsamples = 0, 0
    with open("{}.input0".format(filename)) as input0_f:
        with open("{}.input1".format(filename)) as input1_f:
            with open("{}.label".format(filename)) as label_f:
                for index, (sent1, sent2, label) in tqdm.tqdm(enumerate(zip(input0_f, input1_f, label_f))):
                    sent1=sent1.strip()
                    sent2=sent2.strip()
                    label=label.strip()
                    tokens, prev_tokens = model.encode_tokenized(vocab, sent1, sent2)
                    prediction = model.predict('sentence_classification_head', tokens=tokens, prev_tokens=prev_tokens).argmax().item()
                    prediction_label = label_map[prediction]
                    ncorrect += int(prediction_label == label)
                    nsamples += 1
    print('{} | Accuracy: '.format(lg), float(ncorrect)/float(nsamples))

if __name__ == "__main__":
    args=parse_args()
    vocab = Dictionary.load(args.vocab_path)
    vocab.add_symbol("<mask>")
    for i in range(100):
        vocab.add_symbol(f"<mask_{i}>")
    #vocab.pad_to_multiple_(padding_factor=8)
    lgs=args.lgs.split()
    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt_path)
    model = ElectraEncoderDecoderv6.build_model(state["cfg"].model, src_dict=vocab, tgt_dict=vocab)
    model.register_classification_head(
        "sentence_classification_head",
        num_classes=3,
    )
    model.load_state_dict(state["model"], strict=True)
    print(f"Successfully loading model from {args.ckpt_path}")
    for lg in lgs:
        calculate_accuacy(model, "/mnt/output/XNLI-XLMR/train/test.{}".format(lg), lg=lg)
