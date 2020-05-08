# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Computes correlation betweem BLEURT and human ratings on a test file from WMT."""
import collections
import json

from bleurt import score
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow.compat.v1 as tf

flags = tf.flags
logging = tf.logging
app = tf.app
FLAGS = flags.FLAGS


flags.DEFINE_string("candidate_checkpoint", None,
                    "Path to BLEURT bleurt_checkpoint to benchmark.")

flags.DEFINE_string(
    "candidate_predictions_file", None,
    "Path to WMT-style predictions file. See "
    "http://www.statmt.org/wmt19/metrics-task.html for a description of the "
    "format.")

flags.DEFINE_string("test_file", None,
                    "Path to JSONL ratings file to be used as test data.")


def kendall(pred, ref):
  return stats.kendalltau(pred, ref)[0]


def pearson(pred, ref):
  return stats.pearsonr(pred, ref)[0]


def spearman(pred, ref):
  return stats.spearmanr(pred, ref)[0]


METRICS = {"kendall": kendall, "pearson": pearson, "spearman": spearman}


def eval_checkpoint(export_dir, test_file, results_json=None):

  def _scoring_fun(test_df):
    scorer = score.BleurtScorer(export_dir)
    return scorer.score(test_df.reference, test_df.candidate)

  return run_eval(_scoring_fun, test_file, results_json)


def predict_from_file(path_to_file, test_data_df, filter_newstest=True):
  """Obtains predictions from a file, provided by WMT."""
  tf.logging.info("Evaluating file {}".format(path_to_file))
  col_names = [
      "metric", "lang", "corpus", "system", "segment_id", "prediction", "ens",
      "url"
  ]
  baseline_df = pd.read_csv(
      tf.gfile.Open(path_to_file), sep="\t", names=col_names)
  baseline_df.rename(columns={"rating": "score"}, inplace=True)
  tf.logging.info(baseline_df.head())

  if filter_newstest:
    baseline_df = baseline_df.loc[
        baseline_df["corpus"].str.startswith("newstest"), :]

  join_df = test_data_df.merge(
      baseline_df, how="left", on=["system", "segment_id", "lang"], sort=False)

  # Checks that the join preserves the order of the rows.
  assert join_df["score"].count() == test_data_df["score"].count()
  assert np.all(join_df["score"].to_numpy() == test_data_df["score"].to_numpy())

  predictions = join_df["prediction"]
  n_nulls = predictions.isna().sum()
  if n_nulls > 0:
    tf.logging.warning("Found {} nulls in baseline".format(n_nulls))
  return predictions.tolist()


def eval_prediction_file(prediction_file, test_file, results_json=None):

  def _scoring_fun(test_df):
    assert tf.io.gfile.exists(prediction_file), \
        "Could not find prediction file."
    return predict_from_file(prediction_file, test_df)

  return run_eval(_scoring_fun, test_file, results_json)


def run_eval(scoring_fun, test_file, results_json=None):
  """Computes correlations between BLEURT and human ratings on all pairs of languages."""

  assert tf.io.gfile.exists(test_file), "Could not find test file."
  logging.info("Reading test set.")
  with tf.io.gfile.GFile(test_file, "r") as f:
    test_df = pd.read_json(f, lines=True)
  n_items = len(test_df)
  for col in ["lang", "reference", "candidate", "score"]:
    assert col in test_df.columns, \
        "Field {} not found".format(col)
  logging.info("Read {} examples.".format(n_items))

  logging.info("Obtaining predictions.")
  bleurt_scores = scoring_fun(test_df)
  assert len(bleurt_scores) == n_items
  logging.info("Done.")

  logging.info("Computing the correlations.")
  test_df["bleurt"] = bleurt_scores
  grouped_by_lang = test_df.groupby(by=["lang"])
  results = collections.defaultdict(dict)
  for group_name, group_df in grouped_by_lang:
    logging.info("* {}:".format(group_name))
    predictions = group_df["bleurt"].to_numpy()
    reference = group_df["score"].to_numpy()
    for metric_name in METRICS:
      metric_value = METRICS[metric_name](predictions, reference)
      logging.info("** {}: {}".format(metric_name, metric_value))
      results[group_name][metric_name] = metric_value

  if results_json:
    logging.info("Writing the resutls to disk")
    with tf.io.gfile.GFile(results_json, mode="w+") as out_file:
      out_json = json.dumps(results)
      out_file.write(out_json)

  logging.info("Done.")
  return results


def main(_):
  if FLAGS.candidate_checkpoint:
    eval_checkpoint(FLAGS.candidate_checkpoint, FLAGS.test_file)
  if FLAGS.candidate_predictions_file:
    eval_prediction_file(FLAGS.candidate_predictions_file, FLAGS.test_file)


if __name__ == "__main__":
  tf.app.run(main)
