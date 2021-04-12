# Working with WMT Ratings Data

We provide code to facilitate working with the WMT Metrics
Shared Task and reproduce experiments from our ACL submission. All the scripts are
in the `wmt/` folder.

## Downloading and aggregating the WMT ratings

We found it difficult to work with ratings from the WMT ratings shared task because
the data is spread over several archives. The following command downloads all the
necessary archives and aggregates the ratings in one large [JSONL](http://jsonlines.org/) file.

```
python -m bleurt.wmt.db_builder \
  -target_language="en" \
  -rating_years="2015 2016" \
  -target_file=wmt.jsonl
```
You may use any combination of years from 2015 to 2019.

## Reproducing results from the paper

The script `wmt/benchmark.py` can be used to reproduce results
from our ACL paper. It downloads ratings from the WMT website, postprocesses them,
trains a BLEURT checkpoint and computes the correlation with human ratings.

You may for instance reproduce the results of Table 2 as follows:

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
