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
r"""Tests for finetuning function."""
import os
import tempfile

from bleurt import checkpoint as checkpoint_lib
from bleurt import finetune
from bleurt import score
import tensorflow.compat.v1 as tf

flags = tf.flags
FLAGS = flags.FLAGS


references = [
    "An apple a day keeps the doctor away.",
    "An apple a day keeps the doctor away."
]
candidates = [
    "An apple a day keeps the doctor away.",
    "An apple a day keeps doctors away."
]
ref_scores = [0.910811, 0.771989]


# Utils to get paths to static files.
def get_test_checkpoint():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  ckpt = os.path.join(pkg, "test_checkpoint")
  assert tf.io.gfile.exists(ckpt)
  return ckpt


def get_test_data():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  train_file = os.path.join(pkg, "test_data", "ratings_train.jsonl")
  dev_file = os.path.join(pkg, "test_data", "ratings_dev.jsonl")
  assert tf.io.gfile.exists(train_file)
  assert tf.io.gfile.exists(dev_file)
  return train_file, dev_file


class FinetuneTest(tf.test.TestCase):

  def setUp(self):
    # Saves default FLAG values.
    super(FinetuneTest, self).setUp()
    self._old_flags_val = (FLAGS.init_bleurt_checkpoint, FLAGS.model_dir,
                           FLAGS.num_train_steps, FLAGS.serialized_train_set,
                           FLAGS.serialized_dev_set, FLAGS.init_checkpoint,
                           FLAGS.bert_config_file, FLAGS.vocab_file,
                           FLAGS.max_seq_length, FLAGS.do_lower_case,
                           FLAGS.dynamic_seq_length)

  def tearDown(self):
    # Restores default FLAG values.
    (FLAGS.init_bleurt_checkpoint, FLAGS.model_dir, FLAGS.num_train_steps,
     FLAGS.serialized_train_set, FLAGS.serialized_dev_set,
     FLAGS.init_checkpoint, FLAGS.bert_config_file, FLAGS.vocab_file,
     FLAGS.max_seq_length, FLAGS.do_lower_case, FLAGS.dynamic_seq_length
     ) = self._old_flags_val
    super(FinetuneTest, self).tearDown()

  def test_finetune_and_predict(self):
    checkpoint = get_test_checkpoint()
    train_file, dev_file = get_test_data()

    with tempfile.TemporaryDirectory() as model_dir:
      # Sets new flags.
      FLAGS.init_checkpoint = os.path.join(checkpoint, "variables", "variables")
      FLAGS.bert_config_file = os.path.join(checkpoint, "bert_config.json")
      FLAGS.vocab_file = os.path.join(checkpoint, "vocab.txt")
      FLAGS.do_lower_case = True
      FLAGS.dynamic_seq_length = True
      FLAGS.max_seq_length = 512
      FLAGS.model_dir = model_dir
      FLAGS.num_train_steps = 1
      FLAGS.learning_rate = 0.00000000001
      FLAGS.serialized_train_set = os.path.join(model_dir, "train.tfrecord")
      FLAGS.serialized_dev_set = os.path.join(model_dir, "dev.tfrecord")

      # Runs 1 training step.
      export = finetune.run_finetuning_pipeline(train_file, dev_file)

      # Checks if the pipeline produced a valid BLEURT checkpoint.
      self.assertTrue(tf.io.gfile.exists(export))
      config = checkpoint_lib.read_bleurt_config(export)
      self.assertTrue(type(config), dict)

      # Runs a prediction.
      scorer = score.LengthBatchingBleurtScorer(export)
      scores = scorer.score(references=references, candidates=candidates)
      self.assertLen(scores, 2)
      self.assertAllClose(scores, ref_scores)


if __name__ == "__main__":
  tf.test.main()
