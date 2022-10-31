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
"""Experiment utilities from the Google Language Team.

  Source: https://github.com/google-research/language/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

flags.DEFINE_integer("batch_size", 16, "Batch size.")

flags.DEFINE_integer("eval_batch_size", 16, "Evaluation batch size.")

flags.DEFINE_string("model_dir", None, "Model directory")

flags.DEFINE_integer("tf_random_seed", None, "Random seed for tensorflow")

flags.DEFINE_integer("num_eval_steps", None,
                     "Number of steps to take during evaluation.")

flags.DEFINE_integer("num_train_steps", 25000,
                     "Number of steps to take during training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "Number of steps between checkpoint saves.")

flags.DEFINE_integer("eval_throttle_secs", 0,
                     "Minimum number of seconds to wait between evaluations")

flags.DEFINE_integer("eval_start_delay_secs", 0,
                     "Number of seconds to wait before starting evaluations.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "Max number of checkpoints to keep")

FLAGS = flags.FLAGS


def run_experiment(model_fn,
                   train_input_fn,
                   eval_input_fn,
                   additional_eval_specs=None,
                   exporters=None):
  """Run experiment."""
  run_config = tf_estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      tf_random_seed=FLAGS.tf_random_seed,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max)
  estimator = tf_estimator.Estimator(
      config=run_config, model_fn=model_fn, model_dir=FLAGS.model_dir)
  train_spec = tf_estimator.TrainSpec(
      input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
  assert not additional_eval_specs, (
      "Multiple eval sets are not supported with default experiment runner.")
  eval_spec = tf_estimator.EvalSpec(
      name="default",
      input_fn=eval_input_fn,
      exporters=exporters,
      start_delay_secs=FLAGS.eval_start_delay_secs,
      throttle_secs=FLAGS.eval_throttle_secs,
      steps=FLAGS.num_eval_steps)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf_estimator.train_and_evaluate(
      estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
