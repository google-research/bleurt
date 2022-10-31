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

flags.DEFINE_string("results_json", None,
                    "JSON file where the results will be written.")

flags.DEFINE_integer(
    "sample_size", None,
    "Samples N items from the human ratings without replacement.")

flags.DEFINE_boolean("to_english", False, "To-English language pairs only.")


def kendall(pred, ref):
  return stats.kendalltau(pred, ref)[0]


def pearson(pred, ref):
  return stats.pearsonr(pred, ref)[0]


def spearman(pred, ref):
  return stats.spearmanr(pred, ref)[0]


def grouped_wmt_kendall(df, year=2019, threshold=25):
  """Groups translations by source and computes WMT's Kendall variant."""

  tf.logging.debug("Subset size: {}".format(len(df.index)))
  n_sentences = df["reference"].nunique()
  tf.logging.debug("Number of reference sentences: {}".format(n_sentences))
  df = df.dropna(subset=["bleurt"])
  groups = df.groupby(["reference"])

  agreement, n_pairs, n_skipped = 0, 0, 0
  n_kept_groups, n_skipped_groups = 0, 0
  for _, group_df in groups:

    if len(group_df.index) == 1:
      n_skipped_groups += 1
      continue

    n_kept_groups += 1
    local_agreement, local_n_pairs, local_n_skipped = wmt_kendall(
        group_df, year, threshold, return_counts=True)
    if local_agreement is None or local_n_pairs is None or n_skipped is None:
      # return None
      continue
    agreement += local_agreement
    n_pairs += local_n_pairs
    n_skipped += local_n_skipped

  if n_pairs == 0:
    tf.logging.info(
        "Zero pairs found, skipping. If this behavior is unexpected, "
        "please ensure that raw ratings exist exist.")
    return None

  tf.logging.debug("Found {} agreements among {} pairs".format(
      agreement, n_pairs))
  tf.logging.debug("{} pairs skipped".format(n_skipped))
  tf.logging.debug("{} groups n_kept_groups".format(n_kept_groups))
  tf.logging.debug("{} groups skipped".format(n_skipped_groups))
  return agreement * 1.0 / n_pairs


def wmt_kendall(df, year=2018, threshold=25, return_counts=False):
  """Implement the variant of Kendall Tau used in the WMT metrics shared task."""

  raw_ratings = df["raw_rating"].to_numpy()
  predictions = df["bleurt"].to_numpy()

  finite_ratings = np.isfinite(raw_ratings)
  raw_ratings = raw_ratings[finite_ratings]
  predictions = predictions[finite_ratings]

  if not raw_ratings.size:
    tf.logging.warn("Cannot compute WMT Kendall variant on null raw ratings.")
    if return_counts:
      return None, None, None
    else:
      return None

  if year < 2018:
    ties_matrix = WMT17_TIES_MATRIX
  else:
    ties_matrix = WMT18_TIES_MATRIX
  agreement = 0
  n_pairs = 0
  n_skipped = 0
  n_total = 0
  n_items = len(raw_ratings)
  assert len(predictions) == n_items
  for i in range(n_items - 1):
    for j in range(i + 1, n_items):
      n_total += 1

      if (raw_ratings[i] == raw_ratings[j] or threshold is not None and
          abs(raw_ratings[i] - raw_ratings[j]) < threshold):
        n_skipped += 1
        continue

      if raw_ratings[i] < raw_ratings[j]:
        human_rank = "<"
      elif raw_ratings[i] == raw_ratings[j]:
        human_rank = "="
      elif raw_ratings[i] > raw_ratings[j]:
        human_rank = ">"
      else:
        raise ValueError("Wrong raw_ratings values: {}, {}".format(
            raw_ratings[i], raw_ratings[j]))

      if predictions[i] < predictions[j]:
        pred_rank = "<"
      elif predictions[i] == predictions[j]:
        pred_rank = "="
      elif predictions[i] > predictions[j]:
        pred_rank = ">"
      else:
        raise ValueError("Wrong prediction values: {}, {}".format(
            predictions[i], predictions[j]))

      increment = ties_matrix[(human_rank, pred_rank)]
      assert increment is not None
      agreement += increment
      n_pairs += 1

  assert n_pairs + n_skipped == n_total
  if return_counts:
    return agreement, n_pairs, n_skipped

  tf.logging.debug("Found {} agreements among {} pairs".format(
      agreement, n_pairs))
  return agreement * 1.0 / n_pairs


# The WMT Metrics shared tasks used different weighing schemes in 2017 and 2018.
WMT17_TIES_MATRIX = {
    ("<", "<"): 1,
    ("<", "="): 0,
    ("<", ">"): -1,
    ("=", "<"): None,
    ("=", "="): None,
    ("=", ">"): None,
    (">", "<"): -1,
    (">", "="): 0,
    (">", ">"): 1
}

WMT18_TIES_MATRIX = {
    ("<", "<"): 1,
    ("<", "="): -1,
    ("<", ">"): -1,
    ("=", "<"): None,
    ("=", "="): None,
    ("=", ">"): None,
    (">", "<"): -1,
    (">", "="): -1,
    (">", ">"): 1
}

METRICS = {
    "kendall": kendall,
    "pearson": pearson,
    "spearman": spearman,
    "wmt_da_rr_kendall": grouped_wmt_kendall
}

THRESHOLDS = {2015: 0, 2016: 0, 2017: 0, 2018: 25, 2019: 25, 2020: 25}

# In WMT20, additional systems were removed based on human judgments.
# This is from the Appendix A from the paper
# http://www.statmt.org/wmt20/pdf/2020.wmt-1.77.pdf
# Thanks to Sweeta Agrawal and George Foster for collecting.
WMT_OUTLIERS = {
    2020: {
        "cs-en": ["zlabs-nlp.1149", "CUNI-DocTransformer.1457"],
        "de-en": ["yolo.1052", "zlabs-nlp.1153", "WMTBiomedBaseline.387"],
        "iu-en": ["NiuTrans.1206", "Facebook_AI.729"],
        "ja-en": ["Online-G.1564", "zlabs-nlp.66", "Online-Z.1640"],
        "pl-en": ["zlabs-nlp.1162"],
        "ru-en": ["zlabs-nlp.1164"],
        "ta-en": ["Online-G.1568", "TALP_UPC.192"],
        "zh-en": ["WMTBiomedBaseline.183"],
        "en-cs": ["zlabs-nlp.1151", "Online-G.1555"],
        "en-de": ["zlabs-nlp.179", "WMTBiomedBaseline.388", "Online-G.1556"],
        "en-iu": ["UEDIN.1281", "OPPO.722", "UQAM_TanLe.521"],
        "en-pl": ["Online-Z.1634", "zlabs-nlp.180", "Online-A.1576"],
        "en-ta": ["TALP_UPC.1049", "SJTU-NICT.386", "Online-G.1561"],
        # The list for en-ja was omitted from the appendix; these systems were
        # guessed through trial-and-error to match the scores in table 6.
        "en-ja": ["Online-G.1557", "SJTU-NICT.370"],
    }
}


def eval_checkpoint(export_dir, test_file, results_json=None):
  """Runs evaluation on a BLEURT checkpoint."""

  def _scoring_fun(test_df):
    scorer = score.BleurtScorer(export_dir)
    return scorer.score(
        references=test_df.reference.tolist(),
        candidates=test_df.candidate.tolist())

  return run_eval(_scoring_fun, test_file, results_json)


def eval_tf_export(export_dir, test_file, results_json=None):
  """Runs evaluation on a SavedModel with in-graph tokenization."""
  bleurt_scorer = score.SavedModelBleurtScorer(export_dir)

  def _scoring_fun(test_df):
    scores = bleurt_scorer.score(
        references=test_df.reference, candidates=test_df.candidate)
    return scores

  return run_eval(_scoring_fun, test_file, results_json)


def eval_prediction_file(prediction_file,
                         test_file,
                         results_json=None,
                         wmt_format=True,
                         exclude_sys=None):
  """Runs evaluation on a prediction file, possibly in WMT format."""

  tf.logging.info("Evaluating file {}".format(prediction_file))
  assert tf.io.gfile.exists(prediction_file), "Could not find file."

  def _predict_from_file(test_data_df, filter_newstest=True):

    tf.logging.info("Reading input file.")

    if wmt_format:
      # Reads a file with format WMT 15 to 19.
      col_names = [
          "metric", "lang", "corpus", "system", "segment_id", "prediction",
          "ens", "url"
      ]
      predictions_df = pd.read_csv(
          tf.gfile.Open(prediction_file), sep="\t", names=col_names)
      if filter_newstest:
        predictions_df = predictions_df.loc[
            predictions_df["corpus"].str.startswith("newstest"), :]
    else:
      # Reads a file with free TSV format. The expectation is that the file
      # contains column names, including "system", "segment_id", and "lang".
      predictions_df = pd.read_csv(tf.gfile.Open(prediction_file), sep="\t")
      for col in ["lang", "system", "segment_id"]:
        assert col in predictions_df.columns

    tf.logging.info("Done reading input file.")
    tf.logging.info(predictions_df.head())

    # Joins with ratings data.
    join_df = test_data_df.merge(
        predictions_df,
        how="left",
        on=["system", "segment_id", "lang"],
        sort=False)
    assert join_df["score"].count() == test_data_df["score"].count()
    assert np.all((
        join_df["score"].to_numpy() == test_data_df["score"].to_numpy())
                  | np.isnan(join_df["score"].to_numpy()))

    predictions = join_df["prediction"]
    n_nulls = predictions.isna().sum()
    if n_nulls > 0:
      tf.logging.warning("Found {} nulls in baseline".format(n_nulls))

    return predictions.tolist()

  return run_eval(_predict_from_file, test_file, results_json, exclude_sys)


def run_eval(scoring_fun, test_file, results_json=None, exclude_sys=None):
  """Computes correlations between BLEURT and human ratings on all pairs of languages."""

  assert tf.io.gfile.exists(test_file), "Could not find test file."
  logging.info("Reading test set.")
  with tf.io.gfile.GFile(test_file, "r") as f:
    test_df = pd.read_json(f, lines=True)
  # test_df = test_df[np.isfinite(test_df["score"])]

  if FLAGS.sample_size:
    logging.info("Sampling {} items.".format(str(FLAGS.sample_size)))
    test_df = test_df.sample(n=FLAGS.sample_size, random_state=55555)

  if FLAGS.to_english:
    logging.info("Filtering out non-English.")
    test_df = test_df[test_df["lang"].str.endswith("en")]

  n_items = len(test_df)
  for col in ["year", "lang", "reference", "candidate", "score"]:
    assert col in test_df.columns, \
        "Field {} not found".format(col)
  # Weighting schemes differ across years. Permit only one year at a time.
  assert len(test_df["year"].unique()) == 1
  logging.info("Read {} examples.".format(n_items))

  logging.info("Obtaining predictions.")
  bleurt_scores = scoring_fun(test_df)
  assert len(bleurt_scores) == n_items
  logging.info("Done.")
  test_df["bleurt"] = bleurt_scores

  if exclude_sys:
    tf.logging.info("Excluding systems matching: {}.".format(exclude_sys))
    tf.logging.info("Num rows before: {}.".format(len(test_df.index)))
    test_df = test_df[~test_df["system"].str.match(exclude_sys)]
    tf.logging.info("Num rows after: {}.".format(len(test_df.index)))

  logging.info("Computing correlations.")
  year = test_df["year"].unique()[0]
  grouped_by_lang = test_df.groupby(by=["lang"])
  results = collections.defaultdict(dict)

  for group_name, group_df in grouped_by_lang:
    logging.info("* {}:".format(group_name))

    systems = group_df["system"].unique()
    tf.logging.info("sytems: {}".format(" ".join(systems)))

    # Segment-level correlations.
    predictions = group_df["bleurt"].to_numpy()
    reference = group_df["score"].to_numpy()
    finite_refs = np.isfinite(reference)
    predictions = predictions[finite_refs]
    reference = reference[finite_refs]

    for metric_name in METRICS:
      if metric_name == "wmt_da_rr_kendall":
        metric_value = METRICS[metric_name](group_df, year, THRESHOLDS[year])
      else:
        metric_value = METRICS[metric_name](predictions, reference)
      logging.info("** {}: {}".format(metric_name, metric_value))
      results[group_name][metric_name] = metric_value

    # System-level correlation.
    grouped_by_system = group_df.groupby("system").agg({
        "bleurt": "mean",
        "score": "mean"
    }).reset_index()
    grouped_by_system = grouped_by_system[np.isfinite(
        grouped_by_system["score"])]

    predictions = grouped_by_system["bleurt"].to_numpy()
    reference = grouped_by_system["score"].to_numpy()
    for metric_name in ["kendall", "pearson", "spearman"]:
      metric_value = METRICS[metric_name](predictions, reference)
      logging.info("** sys-{}: {}".format(metric_name, metric_value))
      results[group_name]["sys-" + metric_name] = metric_value

    # System-level, excluding outliers.
    if year not in WMT_OUTLIERS:
      continue
    if group_name in WMT_OUTLIERS[year]:
      outliers = WMT_OUTLIERS[year][group_name]
    else:
      outliers = []

    grouped_by_system_nooutl = grouped_by_system[~grouped_by_system["system"]
                                                 .isin(outliers)]
    predictions = grouped_by_system_nooutl["bleurt"].to_numpy()
    reference = grouped_by_system_nooutl["score"].to_numpy()
    for metric_name in ["kendall", "pearson", "spearman"]:
      metric_value = METRICS[metric_name](predictions, reference)
      logging.info("** sys-{}-nooutl: {}".format(metric_name, metric_value))
      results[group_name]["sys-nooutl-" + metric_name] = metric_value

    # System-level, top 10.
    grouped_by_system_topk = grouped_by_system.nlargest(10, "score")
    predictions = grouped_by_system_topk["bleurt"].to_numpy()
    reference = grouped_by_system_topk["score"].to_numpy()
    for metric_name in ["kendall", "pearson", "spearman"]:
      metric_value = METRICS[metric_name](predictions, reference)
      logging.info("** sys-{}-top10: {}".format(metric_name, metric_value))
      results[group_name]["sys-top10-" + metric_name] = metric_value

  if results_json:
    logging.info("Writing the results to disk")
    with tf.io.gfile.GFile(results_json, mode="w+") as out_file:
      out_json = json.dumps(results)
      out_file.write(out_json)

  logging.info("Done.")
  return results


def main(_):
  if FLAGS.candidate_checkpoint:
    eval_checkpoint(FLAGS.candidate_checkpoint, FLAGS.test_file,
                    FLAGS.results_json)
  if FLAGS.candidate_predictions_file:
    eval_prediction_file(FLAGS.candidate_predictions_file, FLAGS.test_file,
                         FLAGS.results_json)


if __name__ == "__main__":
  tf.app.run(main)
