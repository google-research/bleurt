# BLEURT: a Transfer Learning-Based Metric for Natural Language Generation

BLEURT is an evaluation metric for Natural Language Generation. It takes a pair of sentences as input, a *reference* and a *candidate*, and it returns a score that indicates to what extent the candidate is grammatical and conveys the mearning of the reference. It is comparable to [`sentence-BLEU`](https://en.wikipedia.org/wiki/BLEU) and [`BERTscore`](https://arxiv.org/abs/1904.09675).

BLEURT is a *trained metric*, that is, it is a regression model trained on ratings data. The model is based on [`BERT`](https://arxiv.org/abs/1810.04805). This repository contains all the code necessary to use it and/or fine-tune it for your own applications. BLEURT uses Tensorflow, and it benefits greatly from modern GPUs (it runs on CPU too).

A comprehensive overview of BLEURT can be found in our ACL paper [BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696) and our [blog post](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html).


## Installation

BLEURT runs in Python 3. It relies heavily on `Tensorflow` (>=1.15) and the
library `tf-slim` (>=1.1).
You may install it as follows:

```
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```

You may check your install with unit tests:

```
python -m unittest bleurt.score_test
python -m unittest bleurt.score_not_eager_test
python -m unittest bleurt.finetune_test
```

## Using BLEURT - TL;DR Version

The following commands download the recommended checkpoint and run BLEURT:

```
# Downloads the BLEURT-base checkpoint.
wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip .
unzip bleurt-base-128.zip

# Runs the scoring.
python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=bleurt-base-128
```
The files `bleurt/test_data/candidates` and `references` contain test sentences,
included by default in the BLEURT distribution. The input format is one sentence per line.
You may replace them with your own files. The command outputs one score per sentence pair.


## Using BLEURT - the Long Version

### Command-line tools and APIs

Currently, there are three methods to invoke BLEURT: the command-line tool, the Python API, and the Tensorflow API.

#### Command-line interface

The simplest way to use BLEURT is through command-line, as shown below.

```
python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=bleurt/test_checkpoint \
  -scores_file=scores
```
The files `candidates` and `references` contain one sentence per line (see the folder `test_data` for the exact format). Invoking the command should produce a file `scores` which contains one BLEURT score per sentence pair. Alternatively you may use a [JSONL file](https://jsonlines.org/), as follows:

```
python -m bleurt.score_files \
  -sentence_pairs_file=bleurt/test_data/sentence_pairs.jsonl \
  -bleurt_checkpoint=bleurt/test_checkpoint
```


The flags `bleurt_checkpoint` and `scores_file` are optional. If `bleurt_checkpoint` is not specified, BLEURT will default to the test checkpoint, based on [BERT-Tiny](https://github.com/google-research/bert). Given the modest performance of the model, this is not recommended (more [here](#checkpoints)). If `scores_files` is not specified, BLEURT will use the standard output.

You may also specify the flag `bleurt_batch_size` which determines the number of sentence pairs processed at once by BLEURT. The default value is 16, you may want to increase or decrease it based on the memory available and the presence of a GPU (we typically use 16 when using a MacBook Pro, 100 on a workstation with a GPU).

The following command lists all the other command-line options:

```
python -m bleurt.score_files -helpshort
```

**Apr 9th 2021 Update:** we renamed the command-line tool from `score.py` to `score_files.py`.


#### Python API

BLEURT may be used as a Python library as follows:

```
from bleurt import score

checkpoint = "bleurt/test_checkpoint"
references = ["This is a test."]
candidates = ["This is the test."]

scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)
assert type(scores) == list and len(scores) == 1
print(scores)
```
Here again, BLEURT will default to `BERT-Tiny` if no checkpoint is specified.

BLEURT works both in `eager_mode` (default in TF 2.0) and in a `tf.Session` (TF 1.0), but the latter mode is slower and may be deprecated in the near
future.

**Apr 9th 2021 Update:** we removed the positional arguments; named arguments are now mandatory.


#### Tensorflow API

BLEURT may be embedded in a TF computation graph, e.g., to visualize it
on the Tensorboard while training a model.

The following piece of code shows an example:

```
import tensorflow as tf
# Set tf.enable_eager_execution() if using TF 1.x.

from bleurt import score

references = tf.constant(["This is a test."])
candidates = tf.constant(["This is the test."])

bleurt_ops = score.create_bleurt_ops()
bleurt_out = bleurt_ops(references=references, candidates=candidates)

assert bleurt_out["predictions"].shape == (1,)
print(bleurt_out["predictions"])
```
The crucial part is the call to `score.create_bleurt_ops`, which creates the TF ops.


## Checkpoints

### Overview

A *BLEURT checkpoint* is a self-contained folder that contains a regression model and some information that BLEURT needs to run. BLEURT checkpoints can be downloaded, copy-pasted, and stored anywhere. Furthermore, checkpoints are tunable, which means that they can be fine-tuned on custom ratings data.


BLEURT defaults to the `test` checkpoint, which is light but inaccaurate. We recommend
using [`BLEURT-base-128`](https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip) for results reporting. You may use it as follows:

```
wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip .
unzip bleurt-base-128.zip
python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=bleurt-base-128
```

The [checkpoints page](https://github.com/google-research/bleurt/blob/master/checkpoints.md) provides more information about
how these checkpoints were trained, as well as pointers to additional models.

The checkpoints are not calibrated like BLEU; the results are not in the range [0,1].
Instead, they simulate the human ratings of the [WMT Metrics Shared Task](http://www.statmt.org/wmt19/metrics-task.html), which are standardized per annotator.
We advise to use the metrics for comparison, and recommend against interpreting the absolute values. See [here](https://github.com/google-research/bleurt/issues/1) for more information about BLEURT's calibration.

Each checkpoint is a different model. Thus the results produced by different checkpoints are not directly comparable with each other.


### Fine-tuning checkpoints

You can easily fine-tune BERT or BLEURT checkpoints on your ratings data. The [checkpoints page](https://github.com/google-research/bleurt/blob/master/checkpoints.md) describes how to do so.

## Reproducibility and Training Data

You may find information about how to work with ratings from the [WMT Metrics Shared Task](http://www.statmt.org/wmt19/metrics-task.html) and reproduce results
from [our paper](https://arxiv.org/abs/2004.04696) [here](https://github.com/google-research/bleurt/blob/master/wmt_experiments.md).


## How to Cite

Please cite our ACL paper:

```
@inproceedings{sellam2020bleurt,
  title = {BLEURT: Learning Robust Metrics for Text Generation},
  author = {Thibault Sellam and Dipanjan Das and Ankur P Parikh},
  year = {2020},
  booktitle = {Proceedings of ACL}
}
```
