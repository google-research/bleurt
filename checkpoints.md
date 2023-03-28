# BLEURT Checkpoints

This page provides additional details and links on BLEURT checkpoints.

Important note: each checkpoint is a different model. Thus the results produced by different checkpoints are not directly comparable with each other.

## Overview

A BLEURT checkpoint is a folder that contains a TensorFlow regression model, along with some resources that the model needs to run. BLEURT checkpoints are self-contained, and they can be fine-tuned on ratings data.

Technically, a checkpoint is a [Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators)
with a `bleurt_config.json` file that defines some parmeters for BLEURT. Additionally, the checkpoints below contain additional files required by BERT: a configuation file `bert_config.json` and WordPiece dictionary or SentencePiece model for tokenization.


## The Recommended Checkpoint: BLEURT-20

Currently, the recommended checkpoint is [BLEURT-20](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip). BLEURT-20 is a 32 layers pre-trained Transformer model, [RemBERT](https://openreview.net/forum?id=xpFFI_NtgpW), fine-tuned on:
1. ratings from the WMT Metrics Shared Task (~430K sentence pairs), collected during years 2015 to 2019 of the workshop.
2. synthetic data (~160K sentence pairs), also derived from the WMT corpus. We created "perfect" sentence pairs, by copying the reference into the hypothesis, as well as "catastrophic" ones, by sampling tokens for each language pair randomly.

The details of the architecture and fine-tuning are presented in our [EMNLP paper](https://aclanthology.org/2021.emnlp-main.58/); the architecture is similar to RemBERT-32, with input sequence length 512 instead of 128.


## Distilled Models

To facilitate experimentations, we provide 3 compressed versions of BLEURT-20: [BLEURT-20-D3](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip) (3 layers), [BLEURT-20-D6](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip) (6 layers), and [BLEURT-20-D12](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip) (12 layers). The compression is lossy: smaller models are less accurate. The following table shows their size, performance, and runtime:

| Model | Agreement w. Human, to-En. | Agreement w. Humans from-En. | Parameters | Runtime 1K inputs (mins), no GPU | Runtime 1K inputs (mins), on GPU |
:---- |:---:| :----:|:----:|:----:|:----:|
| [BLEURT-20](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip) | 0.228| 0.520 | 579M | 80.4 | 3.8 |
| [BLEURT-20-D12](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip) | 0.219 | 0.467 | 167M | 24.4 | 1.2 |
| [BLEURT-20-D6](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D6.zip) | 0.211 | 0.429 | 45M | 5.4 | 0.4 |
| [BLEURT-20-D3](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D3.zip) | 0.191 | 0.385 | 30M | 2.7 | 0.2 |

The `Agreement w. Human Ratings` columns report the segment-level correlation with human ratings (Kendall Tau) on WMT Metrics'20 as described in our [EMNLP paper](https://aclanthology.org/2021.emnlp-main.58/). The runtime is reported without any optimization, we expect a 2-10X speedup with [length-based batching](https://github.com/google-research/bleurt/blob/master/README.md#speeding-up-bleurt). We report the the parameter count of the pre-trained models (i.e., without the terminal fully-connected layer). The models and methdology used for distillation are described in our [EMNLP paper](https://aclanthology.org/2021.emnlp-main.58/).

## Historical Checkpoints (English only)

Previously, we released checkpoints based on BERT-English and fine-tuned on [WMT Metrics ratings data from 2015 to 2018](http://www.statmt.org/wmt19/metrics-task.html). We present them below for archival purpose. These checkpoints were trained in three steps: normal BERT pre-training (see [Devlin et al.](https://aclanthology.org/N19-1423/) and [Turc et al.](https://arxiv.org/abs/1908.08962)), pre-training on synthetic ratings, then fine-tuning on the [WMT Metrics](http://www.statmt.org/wmt19/metrics-task.html) database of human ratings, years 2015 to 2018. The general approach is presented in our [ACL paper](https://aclanthology.org/2020.acl-main.704/). Compared to the published results, we used 20k training steps, a batch size of 16, and export every 250 steps.

Name                            | Max #tokens  | Size (#layers, # hidden units)  |
:------------------------------ |:---:| :----:|
[BLEURT-Tiny](https://storage.googleapis.com/bleurt-oss/bleurt-tiny-128.zip) (test)        | 128 | 2-128 |
[BLEURT-Tiny](https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip)                  | 512 | 2-128 |
**[BLEURT-Base](https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip)** (previously recommended)| 128 | 12-768 |
[BLEURT-Base](https://storage.googleapis.com/bleurt-oss/bleurt-base-512.zip)                  | 512 | 12-768 |
[BLEURT-Large](https://storage.googleapis.com/bleurt-oss/bleurt-large-128.zip)                | 128 | 24-1024 |
[BLEURT-Large](https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip)                | 512 | 24-1024 |


The column `max #tokens` specifies the size of BLEURT's input. Internally, the model tokenizes candidate and the reference, concatenates them, then adds 3 special tokens. The field indicates the maximum total number of [WordPiece tokens](https://github.com/google/sentencepiece). If the threshold is exceeded, BLEURT truncates the input.


### Training custom checkpoints

#### From an existing BLEURT checkpoint

BLEURT offers a command-line tool to fine-tune checkpoints on a custom set of ratings.
Currently, we only support fine-tuning the previous generation of checkpoints, based on English BERT (discussed in the previous section).
To illustrate, the following command fine-tunes BERT-tiny on a toy set of examples:

```
python -m bleurt.finetune \
  -init_bleurt_checkpoint=bleurt/test_checkpoint \
  -model_dir=my_new_bleurt_checkpoint \
  -train_set=bleurt/test_data/ratings_train.jsonl \
  -dev_set=bleurt/test_data/ratings_dev.jsonl \
  -num_train_steps=500
```
You may open the files `test_data/ratings_*.jsonl` for example of how the files should be formatted.
Internally, the script tokenizes the JSON sentences, it serializes them into TFRecord files,
and it runs a train/eval loop. It saves the best model and exports it as a BLEURT checkpoint.

If you are a Tensorboard user, you may visualize the training progress:

```
tensorboard --logdir my_new_bleurt_checkpoint
```

The fine-tuning script involves a lot of parameters, such as `batch_size`, `learning_rate`,
`save_checkpoints_steps`, or `max_seq_length`.  You may list them as follows: ```
python finetune.py -helpfull ```. Many of them are borrowed from the
[BERT codebase](https://github.com/google-research/bert).


In general, we warmly recommend using a GPU to fine-tune BERT. Please note that most
commodity GPGPU do not currently have the capacity to fine-tune BERT-large,
more info [here](https://github.com/google-research/bert/blob/master/README.md#out-of-memory-issues).


#### From BERT

If you wish to train a new metric from a "fresh" [BERT checkpoint](http://github.com/google-research/bert)
(that is, not fine-tuned on ratings), you can easily do so. The API is almost the same as fine-tuning from BLEURT:

```
BERT_DIR=bleurt/test_checkpoint
BERT_CKPT=variables/variables
python -m bleurt.finetune \
  -init_checkpoint=${BERT_DIR}/${BERT_CKPT} \
  -bert_config_file=${BERT_DIR}/bert_config.json \  # you may also specify a `sentence_piece_model` if working with RemBERT.
  -vocab_file=${BERT_DIR}/vocab.txt \
  -model_dir=my_new_bleurt_checkpoint \
  -train_set=bleurt/test_data/ratings_train.jsonl \
  -dev_set=bleurt/test_data/ratings_dev.jsonl \
  -num_train_steps=500
```

We also release the checkpoints of BERT "warmed up" with synthetic ratings, as
explained in our [ACL paper](https://arxiv.org/abs/2004.04696). You may find them here:

| Pre-trained Model |
|:-----:|
[BERT-tiny](https://storage.googleapis.com/bleurt-oss/bert-tiny-midtrained.zip) |
[BERT-base](https://storage.googleapis.com/bleurt-oss/bert-base-midtrained.zip) |
[BERT-large](https://storage.googleapis.com/bleurt-oss/bert-large-midtrained.zip) |
