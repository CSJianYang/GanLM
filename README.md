# GanLM: Encoder-Decoder Pre-training with an Auxiliary Discriminator

GanLM is a sequence-to-sequence pre-training model for 
both language generation and understanding tasks.

![picture](https://yuweiyin.com/files/img/2023-07-10-ACL-GanLM.png)

* **Paper**:
  * arXiv: https://arxiv.org/abs/2212.10218
  * ACL 2023: https://aclanthology.org/2023.acl-long.522/
* **Abstract**:

```text
Pre-trained models have achieved remarkable success in natural language processing (NLP). 
However, existing pre-training methods underutilize the benefits of language understanding 
for generation. Inspired by the idea of Generative Adversarial Networks (GANs), we propose 
a GAN-style model for encoder-decoder pre-training by introducing an auxiliary discriminator, 
unifying the ability of language understanding and generation in a single model. Our model, 
named as GanLM, is trained with two pre-training objectives: replaced token detection and 
replaced token denoising. Specifically, given masked source sentences, the generator outputs 
the target distribution and the discriminator predicts whether the target sampled tokens from 
distribution are incorrect. The target sentence is replaced with misclassified tokens to 
construct noisy previous context, which is used to generate the gold sentence. In general, 
both tasks improve the ability of language understanding and generation by selectively using 
the denoising data. Extensive experiments in language generation benchmarks show that GanLM 
with the powerful language understanding capability outperforms various strong pre-trained 
language models (PLMs) and achieves state-of-the-art performance.
```

## Data Preparation

Following the previous work [RoBERTa](https://arxiv.org/pdf/1907.11692.pdf), 
Our English pre-trained model GanLM is trained on 160GB English monolingual data from 
BookCorpus, CC-News, OpenWebText, and CCStories.

In addition, we pre-train GanLM-m with 6TB multilingual data 
as the pioneering work [DeltaLM](https://arxiv.org/pdf/2106.13736.pdf), 
which is a combination of CC100, CCNet, and Wikipedia, covering 100 languages.

All texts are tokenized by [SentencePiece](https://arxiv.org/pdf/1808.06226.pdf) 
(SentencePiece [GitHub Repo](https://github.com/google/sentencepiece)) 
and encoded by the dictionary from [XLM-R](https://arxiv.org/pdf/1911.02116.pdf).

Download the SentencePiece model and tokenization dictionary from [Google Drive](https://drive.google.com/drive/folders/14cf_jLm7Bwz7tb7fd2ZI-SWvdzfQ9aoS?usp=sharing)

* `sentence_piece/spm_en.model`: SentencePiece-model-English
* `sentence_piece/spm_multilingual.model`: SentencePiece-model-Multilingual
* `dictionary/dict_en.txt`: Tokenization-Dictionary-English
* `dictionary/dict_multilingual.txt`: Tokenization-Dictionary-Multilingual


## Model Checkpoints

Download the trained checkpoints from [Google Drive](https://drive.google.com/drive/folders/14cf_jLm7Bwz7tb7fd2ZI-SWvdzfQ9aoS?usp=sharing)

* `model_checkpoint/model_base_en.pt`: GanLM-base-English
* `model_checkpoint/model_base_multilingual.pt`: GanLM-base-Multilingual
* `model_checkpoint/model_base_multilingual_ft100lang.pt`: GanLM-base-Multilingual + Finetune on 100 languages
* `model_checkpoint/model_large_en.pt`: GanLM-large-English
* `model_checkpoint/model_large_multilingual.pt`: GanLM-large-Multilingual


## GanLM Fine-tuning

### Fine-tuning on Generation Task (XSum)

```bash
DATA_DIR="/path/to/data_dir/"  # Required: the directory of the tokenized data file (spm: sentence piece)
PRETRAINED_MODEL_PATH="/path/to/model_file"  # Required: the filepath of the pretrained model checkpoint
bash finetune_generartion.sh "${DATA_DIR}" "${PRETRAINED_MODEL_PATH}"
```

### Fine-tuning on Understanding Task (XNLI)

```bash
DATA_DIR="/path/to/data_dir/"  # Required: the directory of the tokenized data file (spm: sentence piece)
PRETRAINED_MODEL_PATH="/path/to/model_file"  # Required: the filepath of the pretrained model checkpoint
bash finetune_understanding.sh "${DATA_DIR}" "${PRETRAINED_MODEL_PATH}"
```


## License

GanLM is [MIT-licensed](./LICENSE).


## Citation

Please cite as:

```bibtex
@inproceedings{GanLM,
    title = "{G}an{LM}: Encoder-Decoder Pre-training with an Auxiliary Discriminator",
    author = "Yang, Jian  and
      Ma, Shuming  and
      Dong, Li  and
      Huang, Shaohan  and
      Huang, Haoyang  and
      Yin, Yuwei  and
      Zhang, Dongdong  and
      Yang, Liqun  and
      Wei, Furu  and
      Li, Zhoujun",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.522",
    pages = "9394--9412",
}

```
