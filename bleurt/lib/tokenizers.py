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
"""Wrapper classes for various types of tokenization."""

from bleurt.lib import bert_tokenization
import tensorflow.compat.v1 as tf
import sentencepiece as spm

flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS


class Tokenizer(object):
  """Base class for WordPiece and TokenPiece tokenizers."""

  def tokenize(self):
    raise NotImplementedError()

  def tokens_to_id(self):
    raise NotImplementedError()


class WordPieceTokenizer(Tokenizer):
  """Wrapper around BERT's FullTokenizer."""

  def __init__(self, vocab_file, do_lower_case):
    logging.info("Creating WordPiece tokenizer.")
    self.vocab_file = vocab_file
    self.do_lower_case = do_lower_case
    self._tokenizer = bert_tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    logging.info("WordPiece tokenizer instantiated.")

  def tokenize(self, text):
    return self._tokenizer.tokenize(text)

  def convert_tokens_to_ids(self, tokens):
    return self._tokenizer.convert_tokens_to_ids(tokens)


class SentencePieceTokenizer(Tokenizer):
  """Wrapper around SentencePiece tokenizer."""

  def __init__(self, sp_model):
    logging.info("Creating SentencePiece tokenizer.")
    self._sp_model_path = sp_model  + ".model"
    logging.info("Will load model: {}.".format(self._sp_model_path))
    self._sp_model = spm.SentencePieceProcessor()
    self._sp_model.Load(self._sp_model_path)
    self.vocab_size = self._sp_model.GetPieceSize()
    logging.info("SentencePiece tokenizer created.")

  def tokenize(self, text):
    return self._sp_model.EncodeAsPieces(text)

  def convert_tokens_to_ids(self, tokens):
    return [self._sp_model.PieceToId(token) for token in tokens]


def create_tokenizer(vocab_file=None, do_lower_case=None, sp_model=None):
  """Factory function for tokenizers."""
  if vocab_file and do_lower_case is not None:
    return WordPieceTokenizer(vocab_file, do_lower_case)

  elif sp_model:
    logging.info("Creating SentencePiece tokenizer.")
    return SentencePieceTokenizer(sp_model)

  else:
    raise ValueError("Cannot determine the type of Tokenizer to build from "
                     "arguments.")
