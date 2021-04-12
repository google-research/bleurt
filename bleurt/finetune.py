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
r"""Fine-tunes a BERT/BLEURT checkpoint."""
import os

from bleurt import checkpoint as checkpoint_lib
from bleurt import encoding
from bleurt import model
from bleurt.lib import experiment_utils
import tensorflow.compat.v1 as tf


flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS

# Data pipeline.
flags.DEFINE_string("train_set", None,
                    "Path to JSONL file for the training ratings.")

flags.DEFINE_string("dev_set", None, "Path to JSONL file for the dev ratings.")

flags.DEFINE_string(
    "serialized_train_set", None,
    "Target file where the serialized train set will be"
    " created. Will use a temp file if None.")

flags.DEFINE_string(
    "serialized_dev_set", None,
    "Target file where the serialized dev set will be"
    " created. Will use a temp file if None.")

# See model.py and lib/experiment_utils.py for other important flags.


def run_finetuning_pipeline(train_set, dev_set, run_in_lazy_mode=True):
  """Runs the full BLEURT fine-tuning pipeline."""

  if run_in_lazy_mode:
    tf.disable_eager_execution()

  bleurt_params = checkpoint_lib.get_bleurt_params_from_flags_or_ckpt()

  # Preprocessing and encoding for train and dev set.
  logging.info("*** Running pre-processing pipeline for training examples.")
  if FLAGS.serialized_train_set:
    train_tfrecord = FLAGS.serialized_train_set
  else:
    train_tfrecord = train_set + ".tfrecord"
  encoding.encode_and_serialize(
      train_set,
      train_tfrecord,
      vocab_file=bleurt_params["vocab_file"],
      do_lower_case=bleurt_params["do_lower_case"],
      sp_model=bleurt_params["sp_model"],
      max_seq_length=bleurt_params["max_seq_length"])

  logging.info("*** Running pre-processing pipeline for eval examples.")
  if FLAGS.serialized_dev_set:
    dev_tfrecord = FLAGS.serialized_dev_set
  else:
    dev_tfrecord = dev_set + ".tfrecord"
  encoding.encode_and_serialize(
      dev_set,
      dev_tfrecord,
      vocab_file=bleurt_params["vocab_file"],
      do_lower_case=bleurt_params["do_lower_case"],
      sp_model=bleurt_params["sp_model"],
      max_seq_length=bleurt_params["max_seq_length"])

  # Actual fine-tuning work.
  logging.info("*** Running fine-tuning.")
  train_eval_fun = experiment_utils.run_experiment
  model.run_finetuning(train_tfrecord, dev_tfrecord, train_eval_fun)

  # Deletes temp files.
  if not FLAGS.serialized_train_set:
    logging.info("Deleting serialized training examples.")
    tf.io.gfile.remove(train_tfrecord)
  if not FLAGS.serialized_dev_set:
    logging.info("Deleting serialized dev examples.")
    tf.io.gfile.remove(dev_tfrecord)

  # Gets export location.
  glob_pattern = os.path.join(FLAGS.model_dir, "export", "bleurt_best", "*")
  export_dirs = tf.io.gfile.glob(glob_pattern)
  assert export_dirs, "Model export directory not found."
  export_dir = export_dirs[0]

  # Finalizes the BLEURT checkpoint.
  logging.info("Exporting BLEURT checkpoint to {}.".format(export_dir))
  checkpoint_lib.finalize_bleurt_checkpoint(export_dir)

  return export_dir


def main(_):
  if FLAGS.dynamic_seq_length:
    tf.disable_eager_execution()
  assert FLAGS.train_set, "Need to specify a train set."
  assert FLAGS.dev_set, "Need to specify a dev set."
  run_finetuning_pipeline(FLAGS.train_set, FLAGS.dev_set)


if __name__ == "__main__":
  tf.app.run()
