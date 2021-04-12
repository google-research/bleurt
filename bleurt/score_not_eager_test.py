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
"""Tests for scoring functions with TF1-style lazy execution."""
import os

from bleurt import score
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


references = [
    "An apple a day keeps the doctor away.",
    "An apple a day keeps the doctor away."
]
candidates = [
    "An apple a day keeps the doctor away.",
    "An apple a day keeps doctors away."
]
ref_scores = [0.910811, 0.771989]


def get_test_checkpoint():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  ckpt = os.path.join(pkg, "test_checkpoint")
  assert tf.io.gfile.exists(ckpt)
  return ckpt


class ScoreTest(tf.test.TestCase):

  def test_default_bleurt_score(self):
    bleurt = score.BleurtScorer()
    scores = bleurt.score(references=references, candidates=candidates)
    bleurt.close()
    self.assertLen(scores, 2)
    self.assertAllClose(scores, ref_scores)

  def test_tf_bleurt_score_not_eager(self):
    with self.session(graph=tf.Graph()) as session:
      # Creates the TF Graph.
      bleurt_ops = score.create_bleurt_ops()
      bleurt_scores = bleurt_ops(references=references, candidates=candidates)

      # Runs init.
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.tables_initializer())
      session.run(init_op)

      # Computes the BLEURT scores.
      bleurt_out = session.run(bleurt_scores)

    self.assertIn("predictions", bleurt_out)
    self.assertEqual(bleurt_out["predictions"].shape, (2,))
    self.assertAllClose(bleurt_out["predictions"], ref_scores)


if __name__ == "__main__":
  tf.test.main()
