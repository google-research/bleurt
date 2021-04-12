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
r"""Tests for scoring API functionality."""
import os
import tempfile

from bleurt import score_files
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

flags = tf.flags
FLAGS = flags.FLAGS

ref_scores = [0.926138, 0.247466, -0.935921, -1.053069]


def get_test_checkpoint():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  ckpt = os.path.join(pkg, "test_checkpoint")
  assert tf.io.gfile.exists(ckpt)
  return ckpt


def get_test_data():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  sentence_pairs_file = os.path.join(pkg, "test_data", "sentence_pairs.jsonl")
  references_file = os.path.join(pkg, "test_data", "references")
  candidates_file = os.path.join(pkg, "test_data", "candidates")
  assert tf.io.gfile.exists(sentence_pairs_file)
  assert tf.io.gfile.exists(references_file)
  assert tf.io.gfile.exists(candidates_file)
  return sentence_pairs_file, references_file, candidates_file


def get_scores_from_scores_file(scores_file):
  assert tf.io.gfile.exists(scores_file)
  with tf.io.gfile.GFile(FLAGS.scores_file, "r") as file:
    scores = [float(line) for line in file]
  return scores


class ScoreFilesTest(tf.test.TestCase):

  def setUp(self):
    # Saves default FLAG values.
    super(ScoreFilesTest, self).setUp()
    self._old_flags_val = (FLAGS.sentence_pairs_file, FLAGS.candidate_file,
                           FLAGS.reference_file, FLAGS.scores_file,
                           FLAGS.bleurt_checkpoint, FLAGS.bleurt_batch_size)

  def tearDown(self):
    # Restores default FLAG values.
    (FLAGS.sentence_pairs_file, FLAGS.candidate_file, FLAGS.reference_file,
     FLAGS.scores_file, FLAGS.bleurt_checkpoint,
     FLAGS.bleurt_batch_size) = self._old_flags_val
    super(ScoreFilesTest, self).tearDown()

  def test_json_generator_empty(self):
    # Tests AssertionError is raised when specifying an
    # empty file path for generator.
    FLAGS.bleurt_checkpoint = get_test_checkpoint()
    with self.assertRaises(AssertionError):
      generator = score_files._json_generator("")
      score_files.score_files(generator, FLAGS.bleurt_checkpoint)

  def test_text_generator_empty(self):
    # Tests AssertionError is raised when specifying an
    # empty file path for generator.
    FLAGS.bleurt_checkpoint = get_test_checkpoint()
    with self.assertRaises(AssertionError):
      generator = score_files._text_generator("", "")
      score_files.score_files(generator, FLAGS.bleurt_checkpoint)

  def test_score_files_sentence_pairs(self):
    # Tests specifying JSONL file of sentence pairs genereates accurate scores.
    checkpoint = get_test_checkpoint()
    sentence_pairs_file, _, _ = get_test_data()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.scores_file = os.path.join(temp_dir, "scores")
      generator = score_files._json_generator(sentence_pairs_file)
      score_files.score_files(generator, checkpoint)
      self.assertTrue(tf.io.gfile.exists(FLAGS.scores_file))
      scores = get_scores_from_scores_file(FLAGS.scores_file)
      self.assertLen(scores, 4)
      self.assertAllClose(scores, ref_scores)

  def test_score_files_text(self):
    # Tests specifying two text files for candidates
    # and references generates accurate scores.
    checkpoint = get_test_checkpoint()
    _, reference_file, candidate_file = get_test_data()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.scores_file = os.path.join(temp_dir, "scores")
      generator = score_files._text_generator(reference_file, candidate_file)
      score_files.score_files(generator, checkpoint)
      self.assertTrue(tf.io.gfile.exists(FLAGS.scores_file))
      scores = get_scores_from_scores_file(FLAGS.scores_file)
      self.assertLen(scores, 4)
      self.assertAllClose(scores, ref_scores)

  def test_score_diff_sentence_pairs(self):
    # Tests specifying sentence pairs where number of candidates
    # and references lengths differ.
    checkpoint = get_test_checkpoint()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.sentence_pairs_file = os.path.join(temp_dir, "sentence_pairs.jsonl")
      with tf.io.gfile.GFile(FLAGS.sentence_pairs_file, "w+") as sentence_pairs:
        sentence_pairs.write("{\"candidate\": \"sashimi\"}")
      with self.assertRaises(AssertionError):
        generator = score_files._json_generator(FLAGS.sentence_pairs_file)
        score_files.score_files(generator, checkpoint)

  def test_score_diff_text_files(self):
    # Tests specifying two text files where number of candidates
    # and references lengths differ.
    checkpoint = get_test_checkpoint()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.reference_file = os.path.join(temp_dir, "references")
      FLAGS.candidate_file = os.path.join(temp_dir, "candidates")
      with tf.io.gfile.GFile(FLAGS.reference_file, "w+") as references:
        references.write("nigiri\nshrimp tempura\ntonkatsu")
      with tf.io.gfile.GFile(FLAGS.candidate_file, "w+") as candidates:
        candidates.write("ramen\nfish")
      with self.assertRaises(AssertionError):
        generator = score_files._text_generator(FLAGS.reference_file,
                                                FLAGS.candidate_file)
        score_files.score_files(generator, checkpoint)

  def test_sentence_pairs_consume_buffer(self):
    # Tests specifying a number of sentence pairs that
    # exceeds BLEURT batch size, requiring a call to _consume_buffer.
    checkpoint = get_test_checkpoint()
    sentence_pairs_file, _, _ = get_test_data()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.bleurt_batch_size = 1
      FLAGS.scores_file = os.path.join(temp_dir, "scores")
      generator = score_files._json_generator(sentence_pairs_file)
      score_files.score_files(generator, checkpoint)
      scores = get_scores_from_scores_file(FLAGS.scores_file)
      self.assertLen(scores, 4)
      self.assertAllClose(scores, ref_scores)

  def test_text_consume_buffer(self):
    # Tests specifying a number of candidate and reference pairs that
    # exceeds BLEURT batch size, requiring a call to _consume_buffer.
    checkpoint = get_test_checkpoint()
    _, reference_file, candidate_file = get_test_data()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.bleurt_batch_size = 2
      FLAGS.scores_file = os.path.join(temp_dir, "scores")
      generator = score_files._text_generator(reference_file, candidate_file)
      score_files.score_files(generator, checkpoint)
      scores = get_scores_from_scores_file(FLAGS.scores_file)
      self.assertLen(scores, 4)
      self.assertAllClose(scores, ref_scores)

  def test_score_empty_candidate_and_reference_text(self):
    # Tests scoring text files with an empty candidate and reference.
    checkpoint = get_test_checkpoint()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.reference_file = os.path.join(temp_dir, "references")
      FLAGS.candidate_file = os.path.join(temp_dir, "candidates")
      FLAGS.scores_file = os.path.join(temp_dir, "scores")
      with tf.io.gfile.GFile(FLAGS.reference_file, "w+") as references:
        references.write("\n")
      with tf.io.gfile.GFile(FLAGS.candidate_file, "w+") as candidates:
        candidates.write("\n")
      generator = score_files._text_generator(FLAGS.reference_file,
                                              FLAGS.candidate_file)
      score_files.score_files(generator, checkpoint)
      scores = get_scores_from_scores_file(FLAGS.scores_file)
      self.assertLen(scores, 1)
      self.assertAllClose(scores, [0.679957])

  def test_score_empty_reference_and_candidate_pair(self):
    # Tests scoring sentence pairs with empty candidate and empty reference.
    checkpoint = get_test_checkpoint()
    with tempfile.TemporaryDirectory() as temp_dir:
      FLAGS.sentence_pairs_file = os.path.join(temp_dir, "sentence_pairs.jsonl")
      FLAGS.scores_file = os.path.join(temp_dir, "scores")
      with tf.io.gfile.GFile(FLAGS.sentence_pairs_file, "w+") as sentence_pairs:
        sentence_pairs.write("{\"candidate\": \"\", \"reference\": \"\"}")
      generator = score_files._json_generator(FLAGS.sentence_pairs_file)
      score_files.score_files(generator, checkpoint)
      scores = get_scores_from_scores_file(FLAGS.scores_file)
      self.assertLen(scores, 1)
      self.assertAllClose(scores, [0.679957])


if __name__ == "__main__":
  tf.test.main()
