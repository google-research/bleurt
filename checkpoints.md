# BLEURT Checkpoints

This page provides additional details and links about BLEURT checkpoints.


## Overview

A BLEURT checkpoint is a folder that contains a TensorFlow regression model, along with some resources that the model needs to run. Technically, it is a [Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model#savedmodels_from_estimators)
with a `bleurt_config.json` file that defines some parmeters for BLEURT. It also contains two additional files required by BERT, the dictionary `vocab.txt` and the parameters files `bert_config.json`.


Currently, the following six BLEURT checkpoints are available, fine-tuned on [WMT Metrics ratings data from 2015 to 2018](http://www.statmt.org/wmt19/metrics-task.html). They vary on two aspects: the size of the model, and the size of the input. The bigger the model, the more accurately it models human ratings, but the more resources it needs. If you don't know where to start, we recommend using BLEURT-base with 128 tokens.

Name                            | Max #tokens  | Size (#layers, # hidden units)  |
:------------------------------ |:---:| :----:|
[BLEURT-Tiny](https://storage.googleapis.com/bleurt-oss/bleurt-tiny-128.zip) (default)        | 128 | 2-128 |
[BLEURT-Tiny](https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip)                  | 512 | 2-128 |
**[BLEURT-Base](https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip)** (recommended)| 128 | 12-768 |
[BLEURT-Base](https://storage.googleapis.com/bleurt-oss/bleurt-base-512.zip)                  | 512 | 12-768 |
[BLEURT-Large](https://storage.googleapis.com/bleurt-oss/bleurt-large-128.zip)                | 128 | 24-1024 |
[BLEURT-Large](https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip)                | 512 | 24-1024 |

For instance, you may use BLEURT-tiny-512 as follows:

```
wget https://storage.googleapis.com/bleurt-oss/bleurt-tiny-512.zip .
unzip bleurt-tiny-512.zip
python -m bleurt.score \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=bleurt-tiny-512
```

The column `max #tokens` specifies the size of BLEURT's input. Internally, the model tokenizes candidate and the reference, concatenates them, then adds 3 special tokens. The field indicates the maximum total number of [WordPiece tokens](https://github.com/google/sentencepiece). If the threshold is exceeded, BLEURT truncates the input.


Please note that each checkpoint is a different model. Thus the results produced by different checkpoints are not directly comparable with each other.

In generally, the larger the checkpoints the more accurate the ratings. The following table compares BLEURT's performance to that of the best approaches available at the time of writing on the [WMT Metrics shared task 2019](http://www.statmt.org/wmt19/). We report the Kendall Tau between the metrics and human ratings, higher is better.



Model                  | Max #tokens| de-en| fi-en| gu-en|kk-en | lt-en|ru-en | zh-en| **Average** |
:--------------------- |:----------:|:----:|:----:|:----:|-----:|:----:|:----:|:----:|:-----:|
YiSi-SRL               |            | 26.3 | 27.8 | 26.6 | 36.9 | 30.9 | 25.3 | 38.9 | **30.4** |
BERTscore (BERT-large) |            | 26.2 | 27.6 | 25.8 | 36.9 | 30.8 | 25.2 | 37.5 | **30.0** |
BLEU                   |            | 19.4 | 20.6 | 17.3 | 30.0 | 23.8 | 19.4 | 28.7 | **22.7** |
BLEURT-tiny              |   128      | 25.4 | 25.7 | 23.7 | 36.3 | 29.0 | 23.3 | 35.0 | **28.3** |
BLEURT-tiny              |   512      | 26.1 | 26.0 | 24.5 | 36.7 | 29.4 | 23.7 | 36.1 | **28.9** |
BLEURT-base              |   128      | 31.3 | 31.4 | 28.1 | 39.6 | 35.4 | 28.4 | 41.7 | **33.7** |
BLEURT-base              |   512      | 31.0 | 31.2 | 28.3 | 39.1 | 35.2 | 28.0 | 41.7 | **33.5** |
BLEURT-large             |   128      | 31.4 | 31.9 | 28.1 | 39.8 | 35.4 | 28.4 | 42.4 | **33.9** |
BLEURT-large             |   512      | 31.3 | 31.8 | 27.7 | 39.2 | 34.8 | 28.7 | 43.4 | **33.9** |


Those checkpoints were trained in three steps: normal BERT pre-training (see [Devlin et al.](https://arxiv.org/abs/1810.04805) and [Turc et al.](https://arxiv.org/abs/1908.08962)), pre-training on synthetic ratings, then fine-tuning on the [WMT Metrics](http://www.statmt.org/wmt19/metrics-task.html) database of human ratings, years 2015 to 2018. The general approach is presented in our [paper](https://arxiv.org/abs/2004.04696). Compared to the published results, we used 20k training steps, a batch size of 16, and export every 250 steps.


### More about runtime and memory
Three parameters control BLEURT's runtime and memory footprint: the size of the
model, the size of the input, and the batch size. The larger the model, the more resources
it needs. The batch size controls the trade-off between memory and runtime.
In general, the model benefits greatly from GPUs, but it can also run on a CPU.

If you do not have access to a GPU or the memory on your GPU is small,
we recommend using the smaller models and lowering the batch size, at least for development.
Batch size 16 is a good place to start.

For reference, the table below presents BLEURT's runtime on a laptop with no GPU, using batch size 16.

Model                    | Max #tokens| 1k examples (mins) | 2k examples (mins) | 5k examples (mins) |
:---------------------   |:----------:|:----:|:----:|:----:|
BLEURT-tiny              |   128      |  0.1 | 0.1 | 0.3   |
BLEURT-tiny              |   512      |  0.2 | 0.4 | 1.1   |
BLEURT-base              |   128      | 2.0  | 4.1 | 10.1  |
BLEURT-base              |   512      | 12.1 | 23.6 | 57.8 |

We used a MacBook Pro with a 2.2 GHz 6-core Intel Core i7 and 32GB main memory.


### Fine-tuning BLEURT checkpoints

BLEURT offers a command-line tool to fine-tune BLEURT on a custom set of ratings.
To illustrate, the following command fine-tunes BERT-tiny on a toy set of ratings:

```
python -m bleurt.finetune \
  -init_bleurt_checkpoint=bleurt/test_checkpoint \
  -model_dir=my_new_bleurt_checkpoint \
  -train_set=bleurt/test_data/ratings_train.jsonl \
  -dev_set=bleurt/test_data/ratings_dev.jsonl \
  -num_train_steps=500
```
You may open the files `test_data/ratings_*.jsonl` for example of how the files should be formattted.
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
commodity GPGPU do not currently have the capacity to fine-tune BERT/BLEURT-large,
more info [here](https://github.com/google-research/bert/blob/master/README.md#out-of-memory-issues).


### Training directly from BERT

If you wish to train a new metric from a "fresh" [BERT checkpoint](http://github.com/google-research/bert)
(that is, not fine-tuned on ratings), you can easily do so. The API is almost the same as fine-tuning from BLEURT:

```
BERT_DIR=bleurt/test_checkpoint
BERT_CKPT=variables/variables
python -m bleurt.finetune \
  -init_checkpoint=${BERT_DIR}/${BERT_CKPT} \
  -bert_config_file=${BERT_DIR}/bert_config.json \
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
