# Working with WMT Ratings Data

We provide code to facilitate working with the WMT Metrics
Shared Task and reproduce experiments from our ACL submission. All the scripts are
in the `wmt/` folder.

## Downloading and aggregating the WMT ratings

We found it sometimes difficult to work with ratings from the WMT ratings shared task because
the data is spread over several archives. The following command downloads all the
necessary archives and aggregates the ratings in one large [JSONL](http://jsonlines.org/) file.

```
python -m bleurt.wmt.db_builder \
  -target_language="en" \
  -rating_years="2015 2016" \
  -target_file=wmt.jsonl
```
You may use any combination of years from 2015 to 2019.

## EMNLP Paper

We release a subset of models used in Table 1 of our [EMNLP paper](https://arxiv.org/abs/2110.06341) below:

* [Teacher (RemBERT-32)](https://storage.googleapis.com/bleurt-oss-21/rembert-32.zip)
* [RemBERT-3 distilled on WMT and Wikipedia](https://storage.googleapis.com/bleurt-oss-21/rembert-3.zip)
* [RemBERT-6 distilled on WMT and Wikipedia](https://storage.googleapis.com/bleurt-oss-21/rembert-6.zip)
* [RemBERT-12 distilled on WMT and Wikipedia](https://storage.googleapis.com/bleurt-oss-21/rembert-12.zip)
* [RemBERT-12 distilled on WMT and Wikipedia, Germanic (cluster 1)](https://storage.googleapis.com/bleurt-oss-21/rembert-12-germanic.zip)
* [RemBERT-12 distilled on WMT and Wikipedia, Romance (cluster 2)](https://storage.googleapis.com/bleurt-oss-21/rembert-12-romance.zip)
* [RemBERT-12 distilled on WMT and Wikipedia, Indo-Iranian-Tamil (cluster 3)](https://storage.googleapis.com/bleurt-oss-21/rembert-12-indo-iranian-ta.zip)
* [RemBERT-12 distilled on WMT and Wikipedia, Slavic-Finno-Ugric-Kazakh-Turkish.zip (cluster 4)](https://storage.googleapis.com/bleurt-oss-21/rembert-12-slavic%2Bfinn-ugr-kk-tr.zip)
* [RemBERT-12 distilled on WMT and Wikipedia, Sino-Tibetan-Japanases.zip (cluster 5)](https://storage.googleapis.com/bleurt-oss-21/rembert-12-sino-tib-jap.zip)


## ACL Paper (English Only)

The [checkpoints page](https://github.com/google-research/bleurt/blob/master/checkpoints.md) lists models that are similar to those trained for our [2020 ACL paper](https://arxiv.org/abs/2004.04696).

The script `wmt/benchmark.py` can be used to re-trained them from scratch. It downloads ratings from the WMT website, postprocesses them,
trains a BLEURT checkpoint and computes the correlation with human ratings.

You may for instance reproduce the results of Table 2 of the paper as follows:

```
BERT_DIR=bleurt/test_checkpoint
BERT_CKPT=variables/variables
python -m bleurt.wmt.benchmark \
 -train_years="2015 2016" \
 -test_years="2017" \
 -dev_ratio=0.1 \
 -model_dir=bleurt_model \
 -results_json=results.json \
 -init_checkpoint=${BERT_DIR}/${BERT_CKPT} \
 -bert_config_file=${BERT_DIR}/bert_config.json \
 -vocab_file=${BERT_DIR}/vocab.txt \
 -do_lower_case=True \
 -num_train_steps=20000
```
For years 2018 and 2019, the flag `average_duplicates_on_test` must be set
to `False` for a direct comparison with results from the paper. This flag
enables averaging different ratings for each distinct reference-candidate pair,
which the organizers of the WMT shared task started doing in 2018.

The exact correlations will probably be different from those
reported in the paper because of differences in setup and initialization
(expect differences between 0.001 and 0.1).
