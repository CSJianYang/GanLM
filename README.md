# GanLM: Encoder-Decoder Pre-training with an Auxiliary Discriminator

GanLM is a sequence-to-sequence pre-training model for 
both language generation and understanding tasks.

![picture](https://yuweiyin.com/files/img/2023-07-10-ACL-GanLM.png)

* **Paper**:
  * arXiv: https://arxiv.org/abs/2212.10218
  * ACL 2023: Accepted; to be published in July 2023
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

## Fine-tuning on Generation Task

* Abstractive Text Summarization Xsum dataset

```bash
bash finetune_generartion.sh
```

## Fine-tuning on Understanding Task

* XNLI-translation-train-all

```bash
bash finetune_understanding.sh
```

## License

GanLM is [MIT-licensed](./LICENSE).


## Citation

Please cite as:

```bibtex
@article{GanLM,
  title={GanLM: Encoder-Decoder Pre-training with an Auxiliary Discriminator},
  author={Yang, Jian and Ma, Shuming and Dong, Li and Huang, Shaohan and Huang, Haoyang and 
    Yin, Yuwei and Zhang, Dongdong and Yang, Liqun and Li, Zhoujun and Wei, Furu},
  journal={arXiv preprint arXiv:2212.10218},
  year={2022}
}
```
