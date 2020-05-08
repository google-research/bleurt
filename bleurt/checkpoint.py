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
"""Utils to read and write from BLEURT checkpoints."""

import json
import os

import tensorflow.compat.v1 as tf

flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS

CONFIG_FILE = "bleurt_config.json"
WEIGHTS_FILE = os.path.join("variables", "variables")


def get_bleurt_params_from_flags_or_ckpt():
  """Reads BLEURT's parameters from either flags or a json config file."""
  logging.info("Reading BLEURT parameters")

  if FLAGS.init_bleurt_checkpoint:
    logging.info("Reading paramter from BLEURT checkpoint: {}".format(
        FLAGS.init_bleurt_checkpoint))
    config = read_bleurt_config(FLAGS.init_bleurt_checkpoint)
    logging.info("Reads parameters from flags.")
    vocab_file = config["vocab_file"]
    do_lower_case = config["do_lower_case"]
    max_seq_length = config["max_seq_length"]
    bert_config_file = config["bert_config_file"]
    init_checkpoint = config["tf_checkpoint_variables"]

    # The following test gives the user the option to override `max_seq_length`.
    # This should only be used during fine-tuning.
    if FLAGS.max_seq_length:
      logging.warning("Overriding `max_seq_length`. This could have unintended"
                      " consequences.")
      max_seq_length = FLAGS.max_seq_length

  else:
    logging.info("Reads parameters from flags.")
    assert FLAGS.vocab_file, "vocab_file missing"
    vocab_file = FLAGS.vocab_file
    assert FLAGS.do_lower_case, "do_lower_case missing"
    do_lower_case = FLAGS.do_lower_case
    assert FLAGS.max_seq_length, "max_seq_length missing"
    max_seq_length = FLAGS.max_seq_length
    assert FLAGS.bert_config_file, "config_file missing"
    bert_config_file = FLAGS.bert_config_file
    assert FLAGS.init_checkpoint, "init_checkpoint missing"
    init_checkpoint = FLAGS.init_checkpoint

  return {
      "vocab_file": vocab_file,
      "do_lower_case": do_lower_case,
      "max_seq_length": max_seq_length,
      "bert_config_file": bert_config_file,
      "init_checkpoint": init_checkpoint
  }


def read_bleurt_config(path):
  """Reads and checks config file from a BLEURT checkpoint."""
  assert tf.io.gfile.exists(path), \
      "Could not find BLEURT checkpoint {}".format(path)
  config_path = os.path.join(path, CONFIG_FILE)
  assert tf.io.gfile.exists(config_path), \
      ("Could not find BLEURT config file {}. Are you sure {}"
       " is a valid checkpoint?").format(config_path, path)
  logging.info("Config file found, reading.")

  with tf.io.gfile.GFile(config_path, "r") as f:
    raw_config = f.read()
  bleurt_config = json.loads(raw_config)
  logging.info("Will load checkpoint {}".format(bleurt_config["name"]))

  logging.info("Performs basic checks...")
  for k in bleurt_config:
    v = bleurt_config[k]
    logging.info("... {}:{}".format(k, v))
    if not isinstance(v, str):
      continue
    if k.endswith("_file") or k.endswith("_dir"):
      fname = os.path.join(path, bleurt_config[k])
      assert tf.io.gfile.exists(fname), "File {} missing.".format(fname)
      bleurt_config[k] = fname

  bleurt_config["chkpt_dir"] = path
  bleurt_config["tf_checkpoint_variables"] = os.path.join(path, WEIGHTS_FILE)
  return bleurt_config


def finalize_bleurt_checkpoint(tf_export_path):
  """Makes a BLEURT checkpoint from A TF Estimator export."""
  logging.info("Finalizing BLEURT checkpoint.")
  assert tf.io.gfile.exists(tf_export_path), "SavedModel export not found!"

  bleurt_params = get_bleurt_params_from_flags_or_ckpt()
  vocab_file = os.path.join(tf_export_path, "vocab.txt")
  tf.io.gfile.copy(bleurt_params["vocab_file"], vocab_file, overwrite=True)
  bert_config_file = os.path.join(tf_export_path, "bert_config.json")
  tf.io.gfile.copy(
      bleurt_params["bert_config_file"], bert_config_file, overwrite=True)
  bleurt_config = {
      "name": FLAGS.bleurt_checkpoint_name,
      "vocab_file": "vocab.txt",
      "bert_config_file": "bert_config.json",
      "do_lower_case": bleurt_params["do_lower_case"],
      "max_seq_length": bleurt_params["max_seq_length"]
  }
  config_string = json.dumps(bleurt_config)
  config_file = os.path.join(tf_export_path, "bleurt_config.json")
  with tf.io.gfile.GFile(config_file, "w+") as f:
    f.write(config_string)
  logging.info("BLEURT checkpoint created.")
