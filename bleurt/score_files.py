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
"""BLEURT scoring library."""

import itertools
from bleurt import score as score_lib
import pandas as pd
import tensorflow as tf

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "sentence_pairs_file", None,
    "Path to a JSONL file that contains sentence pairs. Each JSON record must "
    "contain the fields `reference` and `candidate`. Overrides `candidate_file`"
    " and `reference_file` flags if specified.")

flags.DEFINE_string(
    "candidate_file", None,
    "Path to candidates text file, with one candidate sentence "
    "per line.")

flags.DEFINE_string(
    "reference_file", None,
    "Path to reference text file, with one reference sentence "
    "per line.")

flags.DEFINE_string(
    "scores_file", None,
    "[optional] Path where the scores will be written. Will use standard "
    "output if unspecified.")

flags.DEFINE_string("bleurt_checkpoint", None,
                    "[optional] Path to BLEURT checkpoint.")

flags.DEFINE_integer("bleurt_batch_size", 16,
                     "Number of sentence pairs per batch.")

flags.DEFINE_integer(
    "read_buffer_size", 100000,
    "Number of lines to read at a time from the input files. "
    "Increase or decrase to ajust memory consumption.")

flags.DEFINE_bool(
    "batch_same_length", False,
    "Enables dynamic batching to speed up inference."
    " [experimental feature]")


def _json_generator(sentence_pairs_file):
  """Yields a generator for iterating from a single JSONL file."""
  assert tf.io.gfile.exists(
      sentence_pairs_file), "Sentence pairs file {} not found".format(
          sentence_pairs_file)
  with tf.io.gfile.GFile(sentence_pairs_file, "r") as pairs_file:
    ratings_df = pd.read_json(pairs_file, lines=True)
    for _, row in ratings_df.iterrows():
      assert row.get("reference") is not None, (
          "Reference sentence not found, are you sure the JSON record "
          "contains a 'reference' field?")
      assert row.get("candidate") is not None, (
          "Candidate sentence not found, are you sure the JSON record "
          "contains a 'candidate' field?")
      yield row.get("reference"), row.get("candidate")


def _text_generator(reference_file, candidate_file):
  """Yields a generator for iterating from two text files."""
  assert tf.io.gfile.exists(
      reference_file), "Reference file {} not found".format(reference_file)
  assert tf.io.gfile.exists(
      candidate_file), "Candidate file {} not found".format(candidate_file)
  with tf.io.gfile.GFile(reference_file, "r") as ref_file:
    with tf.io.gfile.GFile(candidate_file, "r") as cand_file:
      for ref_sentence, cand_sentence in itertools.zip_longest(
          ref_file, cand_file, fillvalue=None):
        assert ref_sentence is not None, (
            "Reference sentence not found, are you sure that the files have "
            "the same size?")
        assert cand_sentence is not None, (
            "Candidate sentence not found, are you sure that the files have "
            "the same size?")
        yield ref_sentence, cand_sentence


def score_files(generator, bleurt_checkpoint):
  """Computes BLEURT scores from a sentence pairs generator.

  Requires that a JSONL file containing both candidate and reference
  sentences or two individual candidate and reference text files be specified,
  with the former overriding the latter if both flags are specified.

  Args:
    generator: A generator yielding reference and candidate sentences.
    bleurt_checkpoint: BLEURT checkpoint used for scoring.
  """
  ref_buffer = []
  cand_buffer = []
  scores_buffer = []

  if not FLAGS.batch_same_length:
    scorer = score_lib.BleurtScorer(bleurt_checkpoint)
  else:
    logging.warning(
        "Enabling same length batching. BEWARE: this is an experimental "
        "feature.")
    scorer = score_lib.LengthBatchingBleurtScorer(bleurt_checkpoint)

  def _consume_buffer():
    scores = scorer.score(
        references=ref_buffer,
        candidates=cand_buffer,
        batch_size=FLAGS.bleurt_batch_size)
    del ref_buffer[:]
    del cand_buffer[:]
    scores_buffer.extend(scores)

  logging.info("Computing BLEURT scores...")
  for ref_sentence, cand_sentence in generator:
    ref_buffer.append(ref_sentence)
    cand_buffer.append(cand_sentence)
    if len(ref_buffer) >= FLAGS.read_buffer_size:
      _consume_buffer()
  if ref_buffer:
    _consume_buffer()
  logging.info("BLEURT scores computed.")

  if FLAGS.scores_file:
    logging.info("Writing to disk.")
    with tf.io.gfile.GFile(FLAGS.scores_file, "w+") as score_file:
      for s in scores_buffer:
        score_file.write("{}\n".format(str(s)))
  else:
    for s in scores_buffer:
      print("{}".format(str(s)))
  logging.info("Done.")


def check_flags_and_score():
  """Creates a file reader and runs model."""
  assert FLAGS.sentence_pairs_file or (
      FLAGS.reference_file and FLAGS.candidate_file
  ), ("Reference and candidate files not found, please specify a JSONL file or "
      "two text files.")
  if FLAGS.sentence_pairs_file:
    sentence_pairs_generator = _json_generator(FLAGS.sentence_pairs_file)
  else:
    sentence_pairs_generator = _text_generator(FLAGS.reference_file,
                                               FLAGS.candidate_file)
  score_files(sentence_pairs_generator, FLAGS.bleurt_checkpoint)


def main(_):
  logging.info("Running BLEURT scoring.")
  check_flags_and_score()


if __name__ == "__main__":
  tf.compat.v1.app.run()
