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
"""Tests for scoring function."""
import os

from bleurt import score
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


references = [
    "An apple a day keeps the doctor away.",
    "An apple a day keeps the doctor away."
]
candidates = [
    "An apple a day keeps the doctor away.",
    "An apple a day keeps doctors away."
]
ref_scores = [0.832904, 0.642367]


def get_test_checkpoint():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  ckpt = os.path.join(pkg, "test_checkpoint")
  assert tf.io.gfile.exists(ckpt)
  return ckpt


class ScoreTest(tf.test.TestCase):

  def test_default_bleurt_score(self):
    bleurt = score.BleurtScorer()
    scores = bleurt.score(references, candidates)
    self.assertLen(scores, 2)
    self.assertAllClose(scores, ref_scores)

  def test_bleurt_nulls(self):
    bleurt = score.BleurtScorer()
    test_references = []
    test_candidates = []
    scores = bleurt.score(test_references, test_candidates)
    self.assertLen(scores, 0)

  def test_bleurt_empty(self):
    bleurt = score.BleurtScorer()
    test_references = [""]
    test_candidates = [""]
    scores = bleurt.score(test_references, test_candidates)
    self.assertLen(scores, 1)

  def test_bleurt_score_with_checkpoint(self):
    checkpoint = get_test_checkpoint()
    bleurt = score.BleurtScorer(checkpoint)
    scores = bleurt.score(references, candidates)
    self.assertLen(scores, 2)
    self.assertAllClose(scores, ref_scores)

  def test_tf_bleurt_score_eager(self):
    # Creates the TF Graph.
    bleurt_ops = score.create_bleurt_ops()
    tfcandidates = tf.constant(candidates)
    tfreferences = tf.constant(references)
    bleurt_out = bleurt_ops(tfreferences, tfcandidates)

    # Computes the BLEURT scores.
    self.assertIn("predictions", bleurt_out)
    self.assertEqual(bleurt_out["predictions"].shape, (2,))
    self.assertAllClose(bleurt_out["predictions"], ref_scores)


if __name__ == "__main__":
  tf.test.main()
