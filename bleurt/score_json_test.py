# coding=utf-8
# Copyright 2026 The Google AI Language Team Authors.
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

"""JSONL sentence inputs must retain their original text."""

import json
import os
import tempfile

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
from bleurt import score
from bleurt import score_files
import numpy as np


class JsonSentencesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    directory = tempfile.TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.path = os.path.join(directory.name, 'sentences.jsonl')

  def write_records(self, records):
    with open(self.path, 'w', encoding='utf-8') as stream:
      for record in records:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')

  @parameterized.named_parameters(
      ('integers', [('00123', '00042')]),
      ('decimals', [('0.10', '1.00'), ('2.500', '-0.0')]),
      ('exponents', [('1e3', '2E-2')]),
      ('special', [('NaN', 'Infinity'), ('-Infinity', 'nan')]),
      ('whitespace', [(' 12 ', '\t34\t')]),
      ('empty', [('', '')]),
      ('unicode', [('café 地球', 'straße é')]),
      ('ordinary', [('A cat sat.', 'The cat sits.')]),
      ('mixed', [('00123', '007'), ('The value is 123.', '42')]),
      ('empty_neighbor', [('42', '07'), ('', '')]),
      ('large_integer', [('9007199254740993', '0009007199254740993')]),
  )
  def test_preserves_sentence_strings(self, pairs):
    self.write_records([
        {'reference': reference, 'candidate': candidate}
        for reference, candidate in pairs
    ])
    actual = list(score_files._json_generator(self.path))
    self.assertEqual(actual, pairs)
    self.assertTrue(
        all(isinstance(value, str) for pair in actual for value in pair))

  def test_extra_columns_do_not_affect_sentences(self):
    self.write_records([
        {'reference': '001', 'candidate': '0.20', 'score': 1.5, 'id': 7}])
    self.assertEqual(
        list(score_files._json_generator(self.path)), [('001', '0.20')])

  @parameterized.parameters('reference', 'candidate')
  def test_missing_field_keeps_error(self, missing):
    record = {'reference': 'sentence', 'candidate': 'sentence'}
    del record[missing]
    self.write_records([record])
    with self.assertRaises(AssertionError):
      list(score_files._json_generator(self.path))

  def test_missing_file_keeps_error(self):
    with self.assertRaises(AssertionError):
      list(score_files._json_generator(self.path))

  def test_invalid_json_keeps_error(self):
    with open(self.path, 'w', encoding='utf-8') as stream:
      stream.write('{invalid JSON')
    with self.assertRaises(ValueError):
      list(score_files._json_generator(self.path))

  def test_numeric_json_values_are_not_stringified(self):
    self.write_records([{'reference': 123, 'candidate': 456}])
    actual = list(score_files._json_generator(self.path))
    self.assertEqual(actual, [(123, 456)])
    self.assertFalse(isinstance(actual[0][0], str))

  def test_parsed_sentences_score_like_direct_input(self):
    pairs = [('00123', '00042'), ('1e3', '2E-2'), ('', '')]
    self.write_records([
        {'reference': reference, 'candidate': candidate}
        for reference, candidate in pairs
    ])
    scorer = score.BleurtScorer()
    self.addCleanup(scorer.close)
    expected = scorer.score(
        references=[p[0] for p in pairs], candidates=[p[1] for p in pairs])
    parsed = list(score_files._json_generator(self.path))
    actual = scorer.score(
        references=[p[0] for p in parsed], candidates=[p[1] for p in parsed])
    np.testing.assert_array_equal(actual, expected)

  def test_file_scoring_with_single_pair_buffers(self):
    pairs = [('00123', '00042'), ('1e3', '2E-2')]
    self.write_records([
        {'reference': reference, 'candidate': candidate}
        for reference, candidate in pairs
    ])
    output = self.path + '.scores'
    scorer = score.BleurtScorer()
    self.addCleanup(scorer.close)
    expected = scorer.score(
        references=[p[0] for p in pairs], candidates=[p[1] for p in pairs],
        batch_size=1)
    with flagsaver.flagsaver(
        scores_file=output, read_buffer_size=1, bleurt_batch_size=1,
        batch_same_length=False):
      score_files.score_files(score_files._json_generator(self.path), None)
    with open(output, encoding='utf-8') as stream:
      actual = [float(line) for line in stream]
    np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
  absltest.main()
