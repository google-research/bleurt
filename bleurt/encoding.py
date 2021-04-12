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
"""Data tokenization, encoding and serialization library."""
import collections

from bleurt.lib import tokenizers
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf


flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS


flags.DEFINE_string(
    "vocab_file", None, "Vocabulary file for WordPiece tokenization. "
    "Overridden if `sentence_piece_model` is specified.")

flags.DEFINE_bool(
    "do_lower_case", None,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models. "
    "Overridden if `sentence_piece_model` is specified.")

flags.DEFINE_string(
    "sentence_piece_model", None,
    "Path to SentencePiece model, without `.model` extension. This flag "
    "will override `vocab_file` and `do_lower_case`.")


def _truncate_seq_pair(tokens_ref, tokens_cand, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  while True:
    total_length = len(tokens_ref) + len(tokens_cand)
    if total_length <= max_length:
      break
    if len(tokens_ref) > len(tokens_cand):
      tokens_ref.pop()
    else:
      tokens_cand.pop()


def encode_example(reference, candidate, tokenizer, max_seq_length):
  """Tokenization and encoding of an example rating.

  Args:
    reference: reference sentence.
    candidate: candidate sentence.
    tokenizer: instance of lib.tokenizers.Tokenizer.
    max_seq_length: maximum length of BLEURT's input after tokenization.

  Returns:
    input_ids: contacatenated token ids of the reference and candidate.
    input_mask: binary mask to separate the input from the padding.
    segment_ids: binary mask to separate the sentences.
  """
  # Tokenizes, truncates and concatenates the sentences, as in:
  #  bert/run_classifier.py
  tokens_ref = tokenizer.tokenize(reference)
  tokens_cand = tokenizer.tokenize(candidate)

  _truncate_seq_pair(tokens_ref, tokens_cand, max_seq_length - 3)

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_ref:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  for token in tokens_cand:
    tokens.append(token)
    segment_ids.append(1)
  tokens.append("[SEP]")
  segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)

  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  return input_ids, input_mask, segment_ids


def serialize_example(reference,
                      candidate,
                      tokenizer,
                      max_seq_length,
                      score=None):
  """Encodes a pair of sentences into a serialized tf.Example.

  Args:
    reference: reference sentence.
    candidate: candidate sentence.
    tokenizer: BERT-style WordPiece tokenizer.
    max_seq_length: maximum length of BLEURT's input after tokenization.
    score: [optional] float that indicates the score to be modelled.

  Returns:
    A serialized tf.Example object.
  """

  def _create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

  def _create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f

  input_ids, input_mask, segment_ids = encode_example(reference, candidate,
                                                      tokenizer, max_seq_length)

  # Creates the TFExample.
  features = collections.OrderedDict()
  features["input_ids"] = _create_int_feature(input_ids)
  features["input_mask"] = _create_int_feature(input_mask)
  features["segment_ids"] = _create_int_feature(segment_ids)

  if score is not None:
    features["score"] = _create_float_feature([score])

  # Serializes and returns.
  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example.SerializeToString()


def serialize_raw_example(reference, candidate):
  """Serializes a pair of sentences into a tf.Example without tokenization.

  Args:
    reference: reference sentence.
    candidate: candidate sentence.

  Returns:
    A serialized tf.Example object.
  """

  def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  if isinstance(reference, str):
    reference = reference.encode("utf-8")
  if isinstance(candidate, str):
    candidate = candidate.encode("utf-8")

  features = collections.OrderedDict()
  features["references"] = _bytes_feature(reference)
  features["candidates"] = _bytes_feature(candidate)
  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  return tf_example.SerializeToString()


def encode_batch(references, candidates, tokenizer, max_seq_length):
  """Encodes a batch of sentence pairs to be fed to a BLEURT checkpoint.

  Args:
    references: list of reference sentences.
    candidates: list of candidate sentences.
    tokenizer: BERT-style WordPiece tokenizer.
    max_seq_length: maximum length of BLEURT's input after tokenization.

  Returns:
    A triplet (input_ids, input_mask, segment_ids), all numpy arrays with type
      np.int64<n_sentences, max_seq_length>.
  """
  encoded_examples = []
  for ref, cand in zip(references, candidates):
    triplet = encode_example(ref, cand, tokenizer, max_seq_length)
    example = np.stack(triplet)
    encoded_examples.append(example)
  stacked_examples = np.stack(encoded_examples)
  assert stacked_examples.shape == (len(encoded_examples), 3, max_seq_length)
  return (stacked_examples[:, 0, :], stacked_examples[:, 1, :],
          stacked_examples[:, 2, :])


def encode_and_serialize(input_file, output_file, vocab_file, do_lower_case,
                         sp_model, max_seq_length):
  """Encodes and serializes a set of ratings in JSON format."""
  assert tf.io.gfile.exists(input_file), "Could not find file."
  logging.info("Reading data...")
  with tf.io.gfile.GFile(input_file, "r") as f:
    examples_df = pd.read_json(f, lines=True)
  for col in ["reference", "candidate", "score"]:
    assert col in examples_df.columns, \
        "field {} not found in input file!".format(col)
  n_records = len(examples_df)
  logging.info("Read {} examples.".format(n_records))

  logging.info("Encoding and writing TFRecord file...")
  tokenizer = tokenizers.create_tokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case, sp_model=sp_model)
  with tf.python_io.TFRecordWriter(output_file) as writer:
    iterator_id, iterator_cycle = 0, max(int(n_records / 10), 1)
    for record in examples_df.itertuples(index=False):
      iterator_id += 1
      if iterator_id % iterator_cycle == 0:
        logging.info("Writing example %d of %d", iterator_id, n_records)
      tf_example = serialize_example(
          record.reference,
          record.candidate,
          tokenizer,
          max_seq_length,
          score=record.score)
      writer.write(tf_example)
  logging.info("Done writing {} tf examples.".format(n_records))
