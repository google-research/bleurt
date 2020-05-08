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
r"""Library to build download and aggregate WMT ratings data.

More info about the datasets: https://www.statmt.org/wmt19/metrics-task.html
"""
import os
import tempfile

from bleurt.wmt import downloaders
import pandas as pd
import tensorflow.compat.v1 as tf

flags = tf.flags
app = tf.app
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist(
    "rating_years", "2015 2016",
    "Years for which the ratings will be downloaded,"
    " between 2015 and 2019. Ex: \"2015 2016 2017\".")

flags.DEFINE_string(
    "target_language", "en",
    "Two letter code for the target language. Ex: \"en\"."
    "A star `*` means all languages.")

flags.DEFINE_string("target_file", "/tmp/wmt_eval.jsonl",
                    "Path to JSONL ratings file to be created.")

flags.DEFINE_bool(
    "average_duplicates", True,
    "Whether all the ratings for the same translation should be averaged "
    "(set to False for the test sets of WMT'18 and '19).")

flags.DEFINE_string(
    "wmt_ratings_file", None,
    "[optional] Path where raw WMT ratings will be stored. Creates a temp "
    "file if None.")

flags.DEFINE_string(
    "temp_directory", None, "[optional] Temporary directory where WMT archives "
    "will be downladed and untared.")


# Post-processing.
flags.DEFINE_float("dev_ratio", None,
                   "Ratio of data allocated to dev set. No split if None")

flags.DEFINE_bool(
    "prevent_leaks", True,
    "Prevent leaks when splitting, i.e., train and dev examples with the same "
    "reference sentence.")

WMT_IMPORTERS = {
    "2015": downloaders.Importer1516,
    "2016": downloaders.Importer1516,
    "2017": downloaders.Importer17,
    "2018": downloaders.Importer18,
    "2019": downloaders.Importer19
}


def create_wmt_dataset(target_file, rating_years, target_language):
  """Creates a JSONL file for a given set of years and a target language."""
  logging.info("*** Downloading ratings data from WMT.")
  assert target_file
  assert not os.path.exists(FLAGS.target_file), \
      "Target file already exists. Aborting."
  assert rating_years, "No target year detected."
  for year in rating_years:
    assert year in WMT_IMPORTERS, "No importer for year {}.".format(year)
  assert target_language
  assert target_language == "*" or len(target_language) == 2, \
      "target_language must be a two-letter language code or `*`."

  with tempfile.TemporaryDirectory(dir=FLAGS.temp_directory) as tmpdir:
    logging.info("Using tmp directory: {}".format(tmpdir))

    n_records_total = 0
    for year in rating_years:
      logging.info("\nProcessing ratings for year {}".format(year))
      tmp_file = os.path.join(tmpdir, "tmp_ratings.json")

      # Builds an importer.
      importer_class = WMT_IMPORTERS[year]
      importer = importer_class(year, tmpdir, tmp_file)
      importer.fetch_files()
      lang_pairs = importer.list_lang_pairs()
      logging.info("Lang pairs found:")
      logging.info(" ".join(lang_pairs))

      for lang_pair in lang_pairs:

        if target_language != "*" and not lang_pair.endswith(target_language):
          logging.info("Skipping language pair {}".format(lang_pair))
          continue

        logging.info("Generating records for {} and language pair {}".format(
            year, lang_pair))
        n_records = importer.generate_records_for_lang(lang_pair)
        logging.info("Imported {} records.".format(str(n_records)))
        n_records_total += n_records

    logging.info("Done processing {} elements".format(n_records_total))
    logging.info("Copying temp file...")
    tf.io.gfile.copy(tmp_file, target_file, overwrite=True)
    logging.info("Done.")


def postprocess(target_file, remove_null_refs=True, average_duplicates=True):
  """Postprocesses a JSONL file of ratings downloaded from WMT."""
  logging.info("\n*** Post-processing WMT ratings {}.".format(target_file))
  assert tf.io.gfile.exists(target_file), "WMT ratings file not found!"
  base_file = target_file + "_raw"
  tf.io.gfile.rename(target_file, base_file, overwrite=True)

  logging.info("Reading and processing wmt data...")
  with tf.io.gfile.GFile(base_file, "r") as f:
    ratings_df = pd.read_json(f, lines=True)
  # ratings_df = ratings_df[["lang", "reference", "candidate", "rating"]]
  ratings_df.rename(columns={"rating": "score"}, inplace=True)

  if remove_null_refs:
    ratings_df = ratings_df[ratings_df["reference"].notnull()]
    assert not ratings_df.empty

  if average_duplicates:
    ratings_df = ratings_df.groupby(by=["lang", "candidate", "reference"]).agg({
        "score": "mean",
    }).reset_index()

  logging.info("Saving clean file.")
  with tf.io.gfile.GFile(target_file, "w+") as f:
    ratings_df.to_json(f, orient="records", lines=True)
  logging.info("Cleaning up old ratings file.")
  tf.io.gfile.remove(base_file)


def _shuffle_no_leak(all_ratings_df, n_train):
  """Splits and shuffles such that there is no train/dev example with the same ref."""

  def is_split_leaky(ix):
    return (
        all_ratings_df.iloc[ix].reference == all_ratings_df.iloc[ix -
                                                                 1].reference)

  assert 0 < n_train < len(all_ratings_df.index)

  # Clusters the examples by reference sentence.
  sentences = all_ratings_df.reference.sample(frac=1, random_state=555).unique()
  sentence_to_ix = {s: i for i, s in enumerate(sentences)}
  all_ratings_df["__sentence_ix__"] = [
      sentence_to_ix[s] for s in all_ratings_df.reference
  ]
  all_ratings_df = all_ratings_df.sort_values(by="__sentence_ix__")
  all_ratings_df.drop(columns=["__sentence_ix__"], inplace=True)

  # Moves the split point until there is no leakage.
  split_ix = n_train
  n_dev_sentences = len(all_ratings_df.iloc[split_ix:].reference.unique())
  if n_dev_sentences == 1 and is_split_leaky(split_ix):
    raise ValueError("Failed splitting data--not enough distinct dev sentences"
                     " to prevent leak.")
  while is_split_leaky(split_ix):
    split_ix += 1
  if n_train != split_ix:
    logging.info("Moved split point from {} to {} to prevent "
                 "sentence leaking".format(n_train, split_ix))

  # Shuffles the train and dev sets separately.
  train_ratings_df = all_ratings_df.iloc[:split_ix].copy()
  train_ratings_df = train_ratings_df.sample(frac=1, random_state=555)
  dev_ratings_df = all_ratings_df.iloc[split_ix:].copy()
  dev_ratings_df = dev_ratings_df.sample(frac=1, random_state=555)
  assert len(train_ratings_df) + len(dev_ratings_df) == len(all_ratings_df)

  # Checks that there is no leakage.
  train_sentences = train_ratings_df.reference.unique()
  dev_sentences = dev_ratings_df.reference.unique()
  tf.logging.info("Using {} and {} unique sentences for train and dev.".format(
      len(train_sentences), len(dev_sentences)))
  assert not bool(set(train_sentences) & set(dev_sentences))

  return train_ratings_df, dev_ratings_df


def _shuffle_leaky(all_ratings_df, n_train):
  """Shuffles and splits the ratings allowing overlap in the ref sentences."""
  all_ratings_df = all_ratings_df.sample(frac=1, random_state=555)
  all_ratings_df = all_ratings_df.reset_index(drop=True)
  train_ratings_df = all_ratings_df.iloc[:n_train].copy()
  dev_ratings_df = all_ratings_df.iloc[n_train:].copy()
  assert len(train_ratings_df) + len(dev_ratings_df) == len(all_ratings_df)
  return train_ratings_df, dev_ratings_df


def shuffle_split(ratings_file,
                  train_file=None,
                  dev_file=None,
                  dev_ratio=.1,
                  prevent_leaks=True):
  """Splits a JSONL WMT ratings file into train/dev."""
  logging.info("\n*** Splitting WMT data in train/dev.")

  assert tf.io.gfile.exists(ratings_file), "WMT ratings file not found!"
  base_file = ratings_file + "_raw"
  tf.io.gfile.rename(ratings_file, base_file, overwrite=True)

  logging.info("Reading wmt data...")
  with tf.io.gfile.GFile(base_file, "r") as f:
    ratings_df = pd.read_json(f, lines=True)

  logging.info("Doing the shuffle / split.")
  n_rows, n_train = len(ratings_df), int((1 - dev_ratio) * len(ratings_df))
  logging.info("Will attempt to set aside {} out of {} rows for dev.".format(
      n_rows - n_train, n_rows))
  if prevent_leaks:
    train_df, dev_df = _shuffle_no_leak(ratings_df, n_train)
  else:
    train_df, dev_df = _shuffle_leaky(ratings_df, n_train)
  logging.info("Created train and dev files with {} and {} records.".format(
      len(train_df), len(dev_df)))

  logging.info("Saving clean file.")
  if not train_file:
    train_file = ratings_file + "_train"
  with tf.io.gfile.GFile(train_file, "w+") as f:
    train_df.to_json(f, orient="records", lines=True)
  if not dev_file:
    dev_file = ratings_file + "_dev"
  with tf.io.gfile.GFile(dev_file, "w+") as f:
    dev_df.to_json(f, orient="records", lines=True)

  logging.info("Cleaning up old ratings file.")
  tf.io.gfile.remove(base_file)


def main(_):
  create_wmt_dataset(FLAGS.target_file, FLAGS.rating_years,
                     FLAGS.target_language)
  postprocess(FLAGS.target_file, average_duplicates=FLAGS.average_duplicates)
  if FLAGS.dev_ratio:
    shuffle_split(
        FLAGS.target_file,
        dev_ratio=FLAGS.dev_ratio,
        prevent_leaks=FLAGS.prevent_leaks)


if __name__ == "__main__":
  app.run(main)
