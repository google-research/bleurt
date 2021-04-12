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
"""BLEURT scoring library."""

import os

from bleurt import checkpoint as checkpoint_lib
from bleurt import encoding
from bleurt.lib import tokenizers
import numpy as np
import tensorflow as tf

logging = tf.compat.v1.logging

DEFAULT_BLEURT_BATCH_SIZE = 16


def _get_default_checkpoint():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  ckpt = os.path.join(pkg, "test_checkpoint")
  assert tf.io.gfile.exists(ckpt), (
      "Default checkpoint not found! Are you sure the install is complete?")
  return ckpt


class Predictor(object):
  """Base class for different types of predictors."""

  def initialize(self):
    pass

  def predict(self, input_dict):
    raise NotImplementedError()

  def close(self):
    pass


class EagerPredictor(Predictor):
  """Runs a BLEURT model in eager mode. Recommended by default."""

  def __init__(self, checkpoint):
    logging.info("Creating Eager Mode predictor.")
    assert tf.executing_eagerly()
    self.checkpoint = checkpoint

  def initialize(self):
    logging.info("Loading model.")
    imported = tf.saved_model.load(self.checkpoint)
    self._bleurt_model_ops = imported.signatures["serving_default"]

  def predict(self, input_dict):
    predictions = self._bleurt_model_ops(
        input_ids=tf.constant(input_dict["input_ids"]),
        input_mask=tf.constant(input_dict["input_mask"]),
        segment_ids=tf.constant(
            input_dict["segment_ids"]))["predictions"].numpy()
    return predictions


class LazyPredictor(Predictor):
  """Runs a BLEURT model in lazy mode, with TF1-style tf.Sessions."""

  def __init__(self, checkpoint):
    logging.info("Creating Lazy Mode predictor.")
    logging.warn("Using Tensorflow Sessions---please call `.close()` when you "
                 "are are done using the BleurtScorer.")
    assert not tf.executing_eagerly()
    self.checkpoint = checkpoint

  def initialize(self):
    """Creates the computation graph and the session."""
    logging.info("Loading model.")

    self._bleurt_graph = tf.Graph()
    with self._bleurt_graph.as_default():

      imported = tf.saved_model.load(self.checkpoint)
      bleurt_model_ops = imported.signatures["serving_default"]
      self._bleurt_ops = bleurt_model_ops(
          input_ids=tf.compat.v1.placeholder(tf.int64, name="input_ids"),
          input_mask=tf.compat.v1.placeholder(tf.int64, name="input_mask"),
          segment_ids=tf.compat.v1.placeholder(tf.int64, name="segment_ids"))

      init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                         tf.compat.v1.tables_initializer())

    self.session = tf.compat.v1.Session(graph=self._bleurt_graph)
    self.session.run(init_op)

    logging.info("Done.")

  def predict(self, input_dict):
    with self._bleurt_graph.as_default():
      bleurt_out = self.session.run(
          self._bleurt_ops, {
              "input_ids:0": input_dict["input_ids"],
              "input_mask:0": input_dict["input_mask"],
              "segment_ids:0": input_dict["segment_ids"],
          })
    return bleurt_out["predictions"]

  def close(self):
    self.session.close()


class PythonPredictor(Predictor):
  """Wrapper around a Python function."""

  def __init__(self, predict_fn):
    tf.logging.info("Creating Python-based predictor.")
    self.predict_fn = predict_fn

  def predict(self, input_dict):
    return self.predict_fn(input_dict)


def _create_predictor(checkpoint=None, predict_fn=None):
  assert checkpoint or predict_fn
  if predict_fn:
    return PythonPredictor(predict_fn)
  if tf.executing_eagerly():
    return EagerPredictor(checkpoint)
  else:
    return LazyPredictor(checkpoint)


# Python API for BLEURT.
class BleurtScorer(object):
  """Class for scoring the BLEURT-similarity between two sentences."""

  def __init__(self, checkpoint=None, predict_fn=None):
    """Initializes the BLEURT model.

    Args:
      checkpoint: BLEURT checkpoint. Will default to BLEURT-tiny if None.
      predict_fn: (optional) prediction function, overrides chkpt_dir. Mostly
        used for testing.

    Returns:
      A BLEURT scorer export.
    """
    if not checkpoint:
      logging.info("No checkpoint specified, defaulting to BLEURT-tiny.")
      checkpoint = _get_default_checkpoint()

    logging.info("Reading checkpoint {}.".format(checkpoint))
    self.config = checkpoint_lib.read_bleurt_config(checkpoint)
    max_seq_length = self.config["max_seq_length"]
    vocab_file = self.config["vocab_file"]
    do_lower_case = self.config["do_lower_case"]
    sp_model = self.config["sp_model"]

    logging.info("Creating BLEURT scorer.")
    self.tokenizer = tokenizers.create_tokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case, sp_model=sp_model)
    self.max_seq_length = max_seq_length
    self._predictor = _create_predictor(checkpoint, predict_fn)
    self._predictor.initialize()
    logging.info("BLEURT initialized.")

  def score(self, *args, references=[], candidates=[], batch_size=None):
    """Scores a collection of references and candidates.

    Args:
      *args: dummy collection of positional arguments.
      references: a list of strings.
      candidates: a list of strings.
      batch_size: number of pairs to process per call to `predict_fn`. A high
        value makes the eval speedier but also more memory-intensive.

    Returns:
      A list of scores.
    """
    assert not args, (
        "The score function does not accept positional arguments. Please "
        "specify the name of the arguments explicitly, i.e., "
        "`score(references=..., candidates=...`)")

    candidates, references = list(candidates), list(references)
    assert len(candidates) == len(references), (
        "The number of candidate sentences must match the number of "
        "reference sentences.")
    if not candidates:
      return []

    if not batch_size:
      batch_size = DEFAULT_BLEURT_BATCH_SIZE

    all_results = []
    for i in range(0, len(candidates), batch_size):
      batch_ref = references[i:i + batch_size]
      batch_cand = candidates[i:i + batch_size]
      input_ids, input_mask, segment_ids = encoding.encode_batch(
          batch_ref, batch_cand, self.tokenizer, self.max_seq_length)
      tf_input = {
          "input_ids": input_ids,
          "input_mask": input_mask,
          "segment_ids": segment_ids
      }
      predict_out = self._predictor.predict(tf_input)
      batch_results = predict_out.tolist()
      all_results.extend(batch_results)

    assert len(all_results) == len(candidates), (
        "Number of predictions does not match sentences: {} vs. {}".format(
            len(all_results), len(candidates)))
    return all_results

  def close(self):
    self._predictor.close()


class LengthBatchingBleurtScorer(BleurtScorer):
  """Experimental implementation of uniform length batching, inspired by BERTscore.

  A good explanation may be found here:
  https://towardsdatascience.com/divide-hugging-face-transformers-training-time-by-2-or-more-21bf7129db9q-21bf7129db9e

  It is not clear to whom the technique should be attributed.
  """

  DEFAULT_SCORE = -10000.0

  def __init__(self, checkpoint=None, predict_fn=None):
    super().__init__(checkpoint, predict_fn)
    assert self.config["dynamic_seq_length"] or predict_fn, (
        "The checkpoint does not support dynamic sequence lengths. Please use "
        "another checkpoint, or use disable same length batching.")

  def score(self, *args, references=[], candidates=[], batch_size=None):
    """Scores a collection of references and candidates.

    Args:
      *args: dummy collection of positional arguments.
      references: a list of strings.
      candidates: a list of strings.
      batch_size: number of pairs to process per call to `predict_fn`. A high
        value makes the eval speedier but also more memory-intensive.

    Returns:
      A list of scores.
    """
    assert not args, (
        "The score function does not accept positional arguments. Please "
        "specify the name of the arguments explicitly, i.e., "
        "`score(references=..., candidates=...`)")

    candidates, references = list(candidates), list(references)
    assert len(candidates) == len(references), (
        "The number of candidate sentences must match the number of "
        "reference sentences.")
    n_items = len(candidates)
    if not candidates:
      return []

    if not batch_size:
      batch_size = DEFAULT_BLEURT_BATCH_SIZE

    # Sorts the sentences by length.
    input_ids, input_mask, segment_ids = encoding.encode_batch(
        references, candidates, self.tokenizer, self.max_seq_length)
    seq_lengths = np.sum(input_mask, axis=1)
    sorted_indices = np.argsort(seq_lengths)
    assert sorted_indices.shape[0] == n_items

    all_results = np.repeat(self.DEFAULT_SCORE, n_items).astype(np.float64)
    batch_lens = []
    for i in range(0, n_items, batch_size):

      # Gets the ids of the examples in the batch.
      batch_indices = sorted_indices[i:i + batch_size]

      # Computes the max sequence length.
      batch_lenghts = seq_lengths[batch_indices]
      batch_max_len = max(max(batch_lenghts), 1)
      batch_lens.append(batch_max_len)

      # Retrieves the examples and truncates the extra padding.
      batch_input_ids = input_ids[batch_indices, :batch_max_len]
      batch_input_mask = input_mask[batch_indices, :batch_max_len]
      batch_segment_ids = segment_ids[batch_indices, :batch_max_len]

      # Runs the inference.
      tf_input = {
          "input_ids": batch_input_ids,
          "input_mask": batch_input_mask,
          "segment_ids": batch_segment_ids
      }
      predict_out = self._predictor.predict(tf_input)

      # Scatters the scores.
      all_results[batch_indices] = predict_out

    assert np.all(all_results > -1000), (
        "Something went wrong while running the dynamic batching scorer.")
    all_results = list(all_results)
    assert len(all_results) == n_items, (
        "Number of predictions does not match sentences: {} vs. {}".format(
            len(all_results), len(candidates)))

    logging.info("Average batch sequence length: {}".format(
        np.mean(batch_lens)))

    return all_results


class SavedModelBleurtScorer:
  """BLEURT class with in-graph string pre-processing."""

  def __init__(self, saved_model=None, serialize_input=True):
    """Initializes BLEURT from a SavedModel."""
    assert saved_model
    self.saved_model_path = saved_model
    self.serialize_input = serialize_input
    logging.info("Reading checkpoint {}.".format(saved_model))
    imported_model = tf.saved_model.load(saved_model)
    self.bleurt_model_ops = imported_model.signatures["serving_default"]
    logging.info("BLEURT initialized.")

  def score(self, *args, references=[], candidates=[], batch_size=None):
    """Scores a collection of references and candidates.

    Args:
      *args: dummy collection of positional arguments.
      references: a list of strings.
      candidates: a list of strings.
      batch_size: number of pairs to process per call to `predict_fn`. A high
        value makes the eval speedier but also more memory-intensive.

    Returns:
      A list of scores.
    """
    assert not args, (
        "The score function does not accept positional arguments. Please "
        "specify the name of the arguments explicitly, i.e., "
        "`score(references=..., candidates=...`)")

    candidates, references = list(candidates), list(references)
    assert len(candidates) == len(references), (
        "The number of candidate sentences must match the number of "
        "reference sentences.")
    if not candidates:
      return []

    if not batch_size:
      batch_size = DEFAULT_BLEURT_BATCH_SIZE

    all_results = []
    for i in range(0, len(candidates), batch_size):
      batch_ref = references[i:i + batch_size]
      batch_cand = candidates[i:i + batch_size]

      if self.serialize_input:
        tfrecords = [
            encoding.serialize_raw_example(r, c)
            for (r, c) in zip(batch_ref, batch_cand)
        ]
        predict_out = self.bleurt_model_ops(
            examples=tf.constant(tfrecords))["predictions"].numpy()
      else:
        predict_out = self.bleurt_model_ops(
            references=tf.constant(batch_ref),
            candidates=tf.constant(batch_cand))["predictions"].numpy()

      batch_results = predict_out.tolist()
      all_results.extend(batch_results)

    assert len(all_results) == len(candidates), (
        "Number of predictions does not match sentences: {} vs. {}".format(
            len(all_results), len(candidates)))
    return all_results

  def close(self):
    pass


# TensorFlow API for BLEURT.
def create_bleurt_preprocessing_ops(tokenizer, max_seq_length):
  """Wraps TF ops for BLEURT preprocessing.

  Args:
    tokenizer: instance of lib.tokenizers.Tokenizer.
    max_seq_length: BERT's max sequence length.

  Returns:
    A function that builds TF ops for BLEURT preprocessing.
  """

  def _py_encode(references, candidates):
    input_ids, input_mask, segment_ids = encoding.encode_batch(
        references, candidates, tokenizer, max_seq_length)
    return input_ids, input_mask, segment_ids

  def bleurt_preprocessing_ops(references, candidates):
    """Builds a computation graph for BLEURT tokenization and encoding."""
    return tf.numpy_function(
        func=_py_encode,
        inp=[references, candidates],
        Tout=(tf.int64, tf.int64, tf.int64))

  return bleurt_preprocessing_ops


def create_bleurt_ops(checkpoint=None, bleurt_model_fn=None):
  """Wraps a TF ops builder for BLEURT.

  Args:
    checkpoint: BLEURT checkpoint.
    bleurt_model_fn: custom BLEURT model ops, overrides chkpt_dir. Used for
      testing.

  Returns:
    A function that builds TF ops for BLEURT.
  """
  if not checkpoint:
    logging.info("No checkpoint specified, defaulting to BLEURT-tiny.")
    checkpoint = _get_default_checkpoint()

  assert bleurt_model_fn or tf.io.gfile.exists(checkpoint), (
      "invalid path '%s'" % checkpoint)

  def bleurt_ops(*args, references=None, candidates=None):
    """Builds computation graph for BLEURT.

    Args:
      *args: dummy positional arguments.
      references: <tf.string>[...] Tensor that contains reference sentences.
      candidates: <tf.string>[...] Tensor that contains candidate sentences.

    Returns:
      A <tf.float>[...] Tensor that contains BLEURT scores.
    """
    logging.info("Creating BLEURT TF Ops...")

    assert not args, (
        "This function does not accept positional arguments. Please "
        "specify the name of the arguments explicitly, i.e., "
        "`bleurt_ops(references=..., candidates=...`).")
    assert references is not None and candidates is not None

    logging.info("Reading info from checkpoint {}".format(checkpoint))
    config = checkpoint_lib.read_bleurt_config(checkpoint)
    max_seq_length = config["max_seq_length"]
    vocab_file = config["vocab_file"]
    do_lower_case = config["do_lower_case"]
    sp_model = config["sp_model"]

    logging.info("Creating tokenizer...")
    tokenizer = tokenizers.create_tokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case, sp_model=sp_model)
    logging.info("Tokenizer created")
    logging.info("Creating BLEURT Preprocessing Ops...")
    bleurt_preprocessing_ops = create_bleurt_preprocessing_ops(
        tokenizer, max_seq_length)
    logging.info("Preprocessing Ops created.")

    logging.info("Loading checkpoint...")
    if not bleurt_model_fn:
      imported = tf.saved_model.load(checkpoint)
      bleurt_model_ops = imported.signatures["serving_default"]
    else:
      bleurt_model_ops = bleurt_model_fn
    logging.info("BLEURT Checkpoint loaded")

    input_ids, input_mask, segment_ids = bleurt_preprocessing_ops(
        references, candidates)
    out = bleurt_model_ops(
        input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    logging.info("BLEURT TF Ops created.")
    return out

  return bleurt_ops
