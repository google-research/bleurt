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
"""Downloads WMT data, runs BLEURT, and compute correlation with human ratings."""

import os

from bleurt import finetune
from bleurt.wmt import db_builder
from bleurt.wmt import evaluator

import tensorflow.compat.v1 as tf

flags = tf.flags
logging = tf.logging
app = tf.app
FLAGS = flags.FLAGS

flags.DEFINE_spaceseplist(
    "train_years", "2015 2016",
    "Years used for train and dev. The flags `dev_ratio` and `prevent_leaks`"
    " specify how to split the data.")

flags.DEFINE_spaceseplist("test_years", "2017", "Years used for test.")

flags.DEFINE_string(
    "data_dir", "/tmp/wmt_data",
    "Directory where the train, dev, and test sets will be "
    "stored.")

flags.DEFINE_bool(
    "average_duplicates_on_test", True,
    "Whether all the ratings for the same translation should be averaged "
    "(should be set to False to replicate the results of WMT 18 and 19).")

flags.DEFINE_string("results_json", None,
                    "[optional] JSON file where the results will be written.")


def run_benchmark():
  """Runs the WMT Metrics Benchmark end-to-end."""
  logging.info("Running WMT Metrics Shared Task Benchmark")

  # Prepares the datasets.
  if not tf.io.gfile.exists(FLAGS.data_dir):
    logging.info("Creating directory {}".format(FLAGS.data_dir))
    tf.io.gfile.mkdir(FLAGS.data_dir)

  train_ratings_file = os.path.join(FLAGS.data_dir, "train_ratings.json")
  dev_ratings_file = os.path.join(FLAGS.data_dir, "dev_ratings.json")
  test_ratings_file = os.path.join(FLAGS.data_dir, "test_ratings.json")

  for f in [train_ratings_file, dev_ratings_file, test_ratings_file]:
    if tf.io.gfile.exists(f):
      logging.info("Deleting existing file: {}".format(f))
      tf.io.gfile.remove(f)
      print("Done.")

  logging.info("\n*** Creating training data. ***")
  db_builder.create_wmt_dataset(train_ratings_file, FLAGS.train_years,
                                FLAGS.target_language)
  db_builder.postprocess(train_ratings_file)
  db_builder.shuffle_split(
      train_ratings_file,
      train_ratings_file,
      dev_ratings_file,
      dev_ratio=FLAGS.dev_ratio,
      prevent_leaks=FLAGS.prevent_leaks)

  logging.info("\n*** Creating test data. ***")
  db_builder.create_wmt_dataset(test_ratings_file, FLAGS.test_years,
                                FLAGS.target_language)
  db_builder.postprocess(
      test_ratings_file, average_duplicates=FLAGS.average_duplicates_on_test)

  # Trains BLEURT.
  logging.info("\n*** Training BLEURT. ***")
  export_dir = finetune.run_finetuning_pipeline(train_ratings_file,
                                                dev_ratings_file)

  # Runs the eval.
  logging.info("\n*** Testing BLEURT. ***")
  if not FLAGS.results_json:
    results_json = os.path.join(FLAGS.data_dir, "results.json")
  else:
    results_json = FLAGS.results_json
  results = evaluator.eval_checkpoint(export_dir, test_ratings_file,
                                      results_json)
  logging.info(results)
  logging.info("\n*** Done. ***")


def main(_):
  run_benchmark()


if __name__ == "__main__":
  tf.app.run(main)
