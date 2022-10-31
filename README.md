# BLEURT: a Transfer Learning-Based Metric for Natural Language Generation

BLEURT is an evaluation metric for Natural Language Generation. It takes a pair of sentences as input, a *reference* and a *candidate*, and it returns a score that indicates to what extent the candidate is fluent and conveys the meaning of the reference. It is comparable to [`sentence-BLEU`](https://en.wikipedia.org/wiki/BLEU), [`BERTscore`](https://arxiv.org/abs/1904.09675), and [`COMET`](https://github.com/Unbabel/COMET).

BLEURT is a *trained metric*, that is, it is a regression model trained on ratings data. The model is based on [`BERT`](https://arxiv.org/abs/1810.04805) and [`RemBERT`](https://arxiv.org/pdf/2010.12821.pdf). This repository contains all the code necessary to use it and/or fine-tune it for your own applications. BLEURT uses Tensorflow, and it benefits greatly from modern GPUs (it runs on CPU too).

An overview of BLEURT can be found in our our [blog post](https://ai.googleblog.com/2020/05/evaluating-natural-language-generation.html). Further details are provided in the ACL paper [BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696) and [our EMNLP paper](http://arxiv.org/abs/2110.06341).


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
python -m unittest bleurt.score_files_test
```

## Using BLEURT - TL;DR Version

The following commands download the recommended checkpoint and run BLEURT:

```
# Downloads the BLEURT-base checkpoint.
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip

# Runs the scoring.
python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=BLEURT-20
```
The files `bleurt/test_data/candidates` and `references` contain test sentences,
included by default in the BLEURT distribution. The input format is one sentence per line.
You may replace them with your own files. The command outputs one score per sentence pair.

**Oct 8th 2021 Update:** we upgraded the recommended checkpoint to BLEURT-20, a more accurate, multilingual model  🎉.


## Using BLEURT - the Long Version

### Command-line tools and APIs

Currently, there are three methods to invoke BLEURT: the command-line interface, the Python API, and the Tensorflow API.

#### Command-line interface

The simplest way to use BLEURT is through command line, as shown below.

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


The flags `bleurt_checkpoint` and `scores_file` are optional. If `bleurt_checkpoint` is not specified, BLEURT will default to a test checkpoint, based on [BERT-Tiny](https://github.com/google-research/bert), which is very light but also very inaccurate (we recommend against using it). If `scores_files` is not specified, BLEURT will use the standard output.


The following command lists all the other command-line options:

```
python -m bleurt.score_files -helpshort
```


#### Python API

BLEURT may be used as a Python library as follows:

```
from bleurt import score

checkpoint = "bleurt/test_checkpoint"
references = ["This is a test."]
candidates = ["This is the test."]

scorer = score.BleurtScorer(checkpoint)
scores = scorer.score(references=references, candidates=candidates)
assert isinstance(scores, list) and len(scores) == 1
print(scores)
```
Here again, BLEURT will default to `BERT-Tiny` if no checkpoint is specified.

BLEURT works both in `eager_mode` (default in TF 2.0) and in a `tf.Session` (TF 1.0), but the latter mode is slower and may be deprecated in the near
future.


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

A *BLEURT checkpoint* is a self-contained folder that contains a regression model and some information that BLEURT needs to run. BLEURT checkpoints can be downloaded, copy-pasted, and stored anywhere. Furthermore, checkpoints are tunable, which means that they can be fine-tuned on custom ratings data.


BLEURT defaults to the `test` checkpoint, which is very inaccaurate. We recommend
using [`BLEURT-20`](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip) for results reporting. You may use it as follows:

```
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=BLEURT-20
```

The [checkpoints page](https://github.com/google-research/bleurt/blob/master/checkpoints.md) provides more information about
how these checkpoints were trained, as well as pointers to smaller models. Additionally, you can fine-tune BERT or existing BLEURT checkpoints on your own ratings data. The [checkpoints page](https://github.com/google-research/bleurt/blob/master/checkpoints.md) describes how to do so.

## Interpreting BLEURT Scores
Different BLEURT checkpoints yield different scores. The currently recommended checkpoint [`BLEURT-20`](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip)
generates scores which are roughly between 0 and 1 (sometimes less than 0, sometimes more than 1), where 0 indicates a random output and 1 a perfect one. As with all automatic metrics, BLEURT scores are noisy. For a robust evaluation of a system's quality, we recommend averaging BLEURT scores across the sentences in a corpus. See the [WMT Metrics Shared Task](http://statmt.org/wmt21/metrics-task.html) for a comparison of metrics on this aspect.


In principle, BLEURT should measure *adequacy*: most of its training data was collected by the WMT organizers who asked to annotators "How much do you agree that the system output adequately expresses the meaning of the reference?" ([WMT Metrics'18](http://www.statmt.org/wmt18/pdf/WMT078.pdf), [Graham et al., 2015](https://minerva-access.unimelb.edu.au/bitstream/handle/11343/56463/Graham_Can-machine-translation.pdf)). In practice however, the answers tend to be very correlated with *fluency* ("Is the text fluent English?"), and we added synthetic noise in the training set which makes the distinction between adequacy and fluency somewhat fuzzy.


## Language Coverage

Currently, [`BLEURT-20`](https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip) was tested on 13 languages: Chinese, Czech, English, French, German, Japanese, Korean, Polish, Portugese, Russian, Spanish, Tamil, Vietnamese (these are languages for which we have held-out ratings data). In theory, it should work for [the 100+ languages of multilingual C4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual), on which [RemBERT](https://arxiv.org/pdf/2010.12821.pdf) was trained.

If you tried any other language and would like to share your experience, either positive or negative, please send us feedback!

## Speeding Up BLEURT

We describe three methods to speed up BLEURT, and how to combine them.

#### Batch size tuning
You may specify the flag `-bleurt_batch_size` which determines the number of sentence pairs processed at once by BLEURT. The default value is 16, you may want to increase or decrease it based on the memory available and the presence of a GPU (we typically use 16 when using a laptop without a GPU, 100 on a workstation with a GPU).


#### Length-based batching
Length-based batching is an optimization which consists in batching examples that have a similar a length and cropping the resulting tensor, to avoid wasting computations on padding tokens. This technique oftentimes results in spectacular speed-ups (typically, ~2-10X). It is described [here](https://towardsdatascience.com/divide-hugging-face-transformers-training-time-by-2-or-more-21bf7129db9q-21bf7129db9e), and it was successfully used by [BERTScore](https://github.com/Tiiiger/bert_score) in the field of learned metrics.

You can enable length-based by specifying `-batch_same_length=True` when calling `score_files` with the command line, or by instantiating a `LengthBatchingBleurtScorer` instead of `BleurtScorer` when using the Python API.


#### Distilled models
We provide pointers to several compressed checkpoints on the [checkpoints page](https://github.com/google-research/bleurt/blob/master/checkpoints.md). These models were obtained by distillation, a *lossy* process, and therefore the outputs cannot be directly compared to those of the original BLEURT model (though they should be strongly correlated).


#### Putting everything together
The following command illustrates how to combine these three techniques, speeding up BLEURT by an order of magnitude (up to 20X with our configuration) on larger files:

```
# Downloads the 12-layer distilled model, which is ~3.5X smaller.
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip .
unzip BLEURT-20-D12.zip

python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_batch_size=100 \            # Optimization 1.
  -batch_same_length=True \           # Optimization 2.
  -bleurt_checkpoint=BLEURT-20-D12    # Optimization 3.
```


## Reproducibility

You may find information about how to work with ratings from the [WMT Metrics Shared Task](http://www.statmt.org/wmt19/metrics-task.html), reproduce results
from [our ACL paper](https://arxiv.org/abs/2004.04696), and a selection of models from [our EMNLP paper](http://arxiv.org/abs/2110.06341) on [this page](https://github.com/google-research/bleurt/blob/master/wmt_experiments.md).


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

The latest model, BLEURT-20, is based on work that led to this follow-up paper:
```
@inproceedings{pu2021learning,
  title = {Learning compact metrics for MT},
  author = {Pu, Amy and Chung, Hyung Won and Parikh, Ankur P and Gehrmann, Sebastian and Sellam, Thibault},
  booktitle = {Proceedings of EMNLP},
  year = {2021}
}
```
