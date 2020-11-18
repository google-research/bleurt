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
import itertools
import os
from pathlib import Path

from bleurt import checkpoint as checkpoint_lib
from bleurt import encoding
from bleurt.lib import tokenization
import tensorflow.compat.v1 as tf

flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS

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

flags.DEFINE_integer("bleurt_batch_size", 100,
                     "Number of sentence pairs per batch.")

flags.DEFINE_bool("average", False, "output mean median score only")

flags.DEFINE_bool("interactive", False, "Launch an interactive shell")


def _get_default_checkpoint():
  pkg = os.path.abspath(__file__)
  pkg, _ = os.path.split(pkg)
  ckpt = os.path.join(pkg, "test_checkpoint")
  assert tf.io.gfile.exists(ckpt), \
      "Default checkpoint not found! Are you sure the install is complete?"
  return ckpt


def _make_eager_predict_fn_from_checkpoint(checkpoint):
  """Creates a prediction function from a checkpoint."""
  assert tf.executing_eagerly()
  imported = tf.saved_model.load_v2(checkpoint)
  bleurt_model_ops = imported.signatures["serving_default"]

  def _predict_fn(input_dict):
    return bleurt_model_ops(
        input_ids=tf.constant(input_dict["input_ids"]),
        input_mask=tf.constant(input_dict["input_mask"]),
        segment_ids=tf.constant(input_dict["segment_ids"])
        )["predictions"].numpy()

  return _predict_fn


def _make_lazy_predict_fn_from_checkpoint(checkpoint):
  """Creates a prediction function from a checkpoint using TF1 Sessions."""
  assert not tf.executing_eagerly()
  logging.warning(
      "Using the old-school tf.Session API. We recommend using the faster "
      "eager implementation by switching TF's `eager execution` mode on.")

  logging.info("Loading model.")
  bleurt_graph = tf.Graph()
  with bleurt_graph.as_default():
    imported = tf.saved_model.load_v2(checkpoint)
    bleurt_model_ops = imported.signatures["serving_default"]
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.tables_initializer())
  logging.info("Done.")

  def _predict_fn(input_dict):
    with tf.Session(graph=bleurt_graph) as session:
      session.run(init_op)
      bleurt_ops = bleurt_model_ops(
          input_ids=tf.constant(input_dict["input_ids"]),
          input_mask=tf.constant(input_dict["input_mask"]),
          segment_ids=tf.constant(input_dict["segment_ids"]))
      bleurt_out = session.run(bleurt_ops)
    return bleurt_out["predictions"]

  return _predict_fn


def _make_predict_fn_from_checkpoint(checkpoint):
  if tf.executing_eagerly():
    return _make_eager_predict_fn_from_checkpoint(checkpoint)
  else:
    return _make_lazy_predict_fn_from_checkpoint(checkpoint)


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
    config = checkpoint_lib.read_bleurt_config(checkpoint)
    max_seq_length = config["max_seq_length"]
    vocab_file = config["vocab_file"]
    do_lower_case = config["do_lower_case"]

    logging.info("Creating BLEURT scorer.")
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    self.max_seq_length = max_seq_length

    if predict_fn:
      self.predict_fn = predict_fn
      logging.info("BLEURT initialized.")
      return

    logging.info("Loading model...")
    self.chkpt_dir = checkpoint
    self.predict_fn = _make_predict_fn_from_checkpoint(checkpoint)
    logging.info("BLEURT initialized.")

  def score(self, references, candidates, batch_size=None):
    """Scores a collection of references and candidates.

    Args:
      references: a list of strings.
      candidates: a list of strings.
      batch_size: number of pairs to process per call to `predict_fn`. A high
        value makes the eval speedier but also more memory-intensive.

    Returns:
      A list of scores.
    """
    if not batch_size:
      batch_size = FLAGS.bleurt_batch_size

    candidates, references = list(candidates), list(references)
    assert len(candidates) == len(references), \
        ("The number of candidate sentences must match the number of "
         "reference sentences.")
    if not candidates:
      return []

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
      predict_out = self.predict_fn(tf_input)
      batch_results = predict_out.tolist()
      all_results.extend(batch_results)

    assert len(all_results) == len(candidates), \
        "Number of predictions does not match sentences: {} vs. {}".format(
            len(all_results), len(candidates))
    return all_results


# TensorFlow API for BLEURT.
def create_bleurt_preprocessing_ops(tokenizer, max_seq_length):
  """Wraps TF ops for BLEURT preprocessing.

  Args:
    tokenizer: WordPiece tokenizer, typically an instance of
      tokenization.FullTokenizer.
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

  assert bleurt_model_fn or tf.io.gfile.exists(checkpoint), \
      "invalid path '%s'" % checkpoint

  def bleurt_ops(references, candidates):
    """Builds computation graph for BLEURT.

    Args:
      references: <tf.string>[...] Tensor that contains reference sentences.
      candidates: <tf.string>[...] Tensor that contains candidate sentences.

    Returns:
      A <tf.float>[...] Tensor that contains BLEURT scores.
    """
    logging.info("Creating BLEURT TF Ops...")

    logging.info("Reading info from checkpoint {}".format(checkpoint))
    config = checkpoint_lib.read_bleurt_config(checkpoint)
    max_seq_length = config["max_seq_length"]
    vocab_file = config["vocab_file"]
    do_lower_case = config["do_lower_case"]

    logging.info("Creating tokenizer...")
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    logging.info("Tokenizer created")
    logging.info("Creating BLEURT Preprocessing Ops...")
    bleurt_preprocessing_ops = create_bleurt_preprocessing_ops(
        tokenizer, max_seq_length)
    logging.info("Preprocessing Ops created.")

    logging.info("Loading checkpoint...")
    if not bleurt_model_fn:
      imported = tf.saved_model.load_v2(checkpoint)
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


def score_files(reference_file: Path, candidate_file: Path, bleurt_checkpoint, do_average=False):
  """Computes BLEURT scores from two files on disk."""
  assert reference_file.exists(), \
      "Reference file {} not found".format(reference_file)
  assert candidate_file.exists(), \
      "Candidate file {} not found".format(candidate_file)

  ref_buffer = []
  cand_buffer = []
  scores_buffer = []
  scorer = BleurtScorer(bleurt_checkpoint)

  def _consume_buffer():
    scores = scorer.score(ref_buffer, cand_buffer, FLAGS.bleurt_batch_size)
    del ref_buffer[:]
    del cand_buffer[:]
    scores_buffer.extend(scores)

  logging.info("Computing BLEURT scores...")

  with reference_file.open("r", encoding='utf-8', errors='ignore') as ref_file:
    with candidate_file.open("r", encoding='utf-8', errors='ignore') as cand_file:
      for ref_sentence, cand_sentence in itertools.zip_longest(
          ref_file, cand_file, fillvalue=None):
        assert ref_sentence is not None, \
            ("Reference sentence not found, are you sure that the files have "
             "the same size?")
        assert cand_sentence is not None, \
            ("Candidate sentence not found, are you sure that the files have "
             "the same size?")
        ref_buffer.append(ref_sentence)
        cand_buffer.append(cand_sentence)
        if len(ref_buffer) >= FLAGS.bleurt_batch_size:
          _consume_buffer()
  if ref_buffer:
    _consume_buffer()
  logging.info("BLEURT scores computed.")

  if do_average:
    import numpy as np
    scores = np.array(scores_buffer)
    print('Mean: %.6f Median: %.6f Total: %d' % (np.mean(scores), np.median(scores), len(scores)))
  else:
    if FLAGS.scores_file:
      logging.info("Writing to disk.")
      with tf.io.gfile.GFile(FLAGS.scores_file, "w+") as score_file:
        for s in scores_buffer:
          score_file.write("{}\n".format(str(s)))
    else:
      for s in scores_buffer:
        print(f"{s:.4f}")
  logging.info("Done.")

def score_interactive(bleurt_checkpoint):
  """Computes BLEURT scores from two files on disk."""
  import readline
  import atexit
  readline.parse_and_bind('set editing-mode emacs')
  histfile = str(Path('~/.python_history').expanduser())
  try:
      readline.read_history_file(histfile)
      # default history len is -1 (infinite), which may grow unruly
      readline.set_history_length(1000)
  except FileNotFoundError:
      pass
  atexit.register(readline.write_history_file, histfile)

  scorer = BleurtScorer(bleurt_checkpoint)

  while True:
    while True:
      ref_line = input('Reference: ').strip()
      if ref_line:
        break
    while True:
      hyp_line = input('Hypothesis: ').strip()
      if hyp_line:
          break
    score = scorer.score([ref_line], [hyp_line], 1)[0]
    print(f'Score: {score:.4f}\n')


def main(_):
  if FLAGS.interactive:

    return score_interactive(FLAGS.bleurt_checkpoint)
  assert FLAGS.reference_file, "Please specify a reference sentences file."
  assert FLAGS.candidate_file, "Please specify a reference sentences file."
  score_files(Path(FLAGS.reference_file), Path(FLAGS.candidate_file),
              FLAGS.bleurt_checkpoint, do_average=FLAGS.average)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.app.run()
