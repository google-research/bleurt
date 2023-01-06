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
"""Downloads ratings data from the WMT Metrics shared task.

More info about the datasets: https://www.statmt.org/wmt19/metrics-task.html
"""
import abc
import collections
import glob
import gzip
import itertools
import json
import os
import re
import shutil
import tarfile
import numpy as np

import six
import tensorflow.compat.v1 as tf

logging = tf.logging
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "wmt_cache_dir", None,
    "[optional] Directory where downloads from the WMT Metrics Shared task"
    " websites will be cached.")

WMT_LOCATIONS = {
    2015: {
        "eval_data": ("DAseg-wmt-newstest2015", "DAseg-wmt-newstest2015.tar.gz",
                      "https://www.scss.tcd.ie/~ygraham/")
    },
    2016: {
        "eval_data": ("DAseg-wmt-newstest2016", "DAseg-wmt-newstest2016.tar.gz",
                      "https://www.scss.tcd.ie/~ygraham/")
    },
    2017: {
        "submissions":
            ("wmt17-metrics-task-no-hybrids", "wmt17-metrics-task-package.tgz",
             "http://ufallab.ms.mff.cuni.cz/~bojar/"),
        "eval_data": ("newstest2017-segment-level-human",
                      "newstest2017-segment-level-human.tar.gz",
                      "https://www.scss.tcd.ie/~ygraham/")
    },
    2018: {
        "submissions":
            ("wmt18-metrics-task-nohybrids", "wmt18-metrics-task-nohybrids.tgz",
             "http://ufallab.ms.mff.cuni.cz/~bojar/wmt18/"),
        "eval_data": ("newstest2018-humaneval", "newstest2018-humaneval.tar.gz",
                      "https://www.scss.tcd.ie/~ygraham/")
    },
    2019: {
        "submissions": ("wmt19-submitted-data-v3",
                        "wmt19-submitted-data-v3-txt-minimal.tgz",
                        "http://ufallab.ms.mff.cuni.cz/~bojar/wmt19/"),
        "eval_data": ("newstest2019-humaneval", "newstest2019-humaneval.tar.gz",
                      "https://www.scss.tcd.ie/~ygraham/")
    }
}


def separate_lang_pair(lang_pair):
  lang_expr = re.compile("([a-z]{2})-([a-z]{2})")
  match = lang_expr.match(lang_pair)
  if match:
    return match.group(1), match.group(2)
  else:
    return None


def postprocess_segment(segment):
  """Various string post-processing necessary to clean the records."""
  # Identifies NULL values.
  if segment == "NO REFERENCE AVAILABLE\n":
    return None
  # Removes trailing \n's.
  segment = segment.strip()
  return segment


@six.add_metaclass(abc.ABCMeta)
class WMTImporter(object):
  """Base class for WMT Importers.

  The aim of WMT importers is to fetch datafiles from the various WMT sources,
  collect information (e.g., list language pairs) and aggregate them into
  one big file.
  """

  def __init__(self, str_year, temp_directory, target_file):
    year = int(str_year[:4])  # Extracts year from the Importer string name.
    self.year = year
    self.location_info = WMT_LOCATIONS[year]
    self.temp_directory = temp_directory
    self.target_file = target_file

  def fetch_files(self):
    """Downloads raw datafiles from various WMT sources."""
    cache = FLAGS.wmt_cache_dir
    if cache and not tf.io.gfile.exists(cache):
      logging.info("Initializing cache {}".format(cache))
      tf.io.gfile.makedirs(cache)

    for file_type in self.location_info:
      folder_name, archive_name, url_prefix = self.location_info[file_type]
      url = url_prefix + archive_name

      if cache:
        cache_path = os.path.join(cache, archive_name)
        download_path = os.path.join(self.temp_directory, archive_name)
        logging.info("Checking cache {}.".format(cache_path))
        if tf.io.gfile.exists(cache_path):
          logging.info("Cache found, copying..")
          tf.io.gfile.copy(cache_path, download_path, overwrite=True)
          logging.info("Done. Untaring...")
          tar = tarfile.open(download_path)
          tar.extractall(path=self.temp_directory)
          tar.close()
          logging.info("Done.")
          continue

      logging.info("File not found in cache.")
      logging.info("Downloading {} from {}".format(folder_name, url))
      _ = tf.keras.utils.get_file(
          archive_name, url, cache_subdir=self.temp_directory, extract=True)
      logging.info("Done")
      if cache:
        logging.info("Writing to cache {}.".format(cache_path))
        tf.io.gfile.copy(download_path, cache_path, overwrite=True)
        logging.info("Done.")

  def list_lang_pairs(self):
    """List all language pairs included in the WMT files for the target year."""
    pass

  def generate_records_for_lang(self, lang):
    """Consolidates all the files for a given language pair and year."""
    pass

  def cleanup(self):
    """Housekeeping--we want to erase all the temp files created."""
    for file_type in self.location_info:
      folder_name, archive_name, _ = self.location_info[file_type]

      # Removes data folder
      folder_path = os.path.join(self.temp_directory, folder_name)
      logging.info("Removing", folder_path)
      try:
        shutil.rmtree(folder_path)
      except OSError:
        logging.info("OS Error--skipping")

      # Removes downloaded archive
      archive_path = os.path.join(self.temp_directory, archive_name)
      logging.info("Removing", archive_path)
      try:
        os.remove(archive_path)
      except OSError:
        logging.info("OS Error--skipping")


class Importer1516(WMTImporter):
  """Importer for years 2015 and 2016."""

  @staticmethod
  def to_json(year, lang, source, reference, candidate, rating, seg_id, system):
    """Converts record to JSON."""
    json_dict = {
        "year": int(year),
        "lang": lang,
        "source": postprocess_segment(source),
        "reference": postprocess_segment(reference),
        "candidate": postprocess_segment(candidate),
        "raw_rating": None,
        "rating": float(rating.strip()),
        "segment_id": seg_id,
        "system": system,
        "n_ratings": None
    }
    return json.dumps(json_dict)

  @staticmethod
  def parse_file_name(fname):
    wmt_pattern = re.compile(r"^DAseg\.newstest([0-9]+)\.[a-z\-]+\.([a-z\-]+)")
    match = re.match(wmt_pattern, fname)
    if match:
      year, lang_pair = int(match.group(1)), match.group(2)
      return year, lang_pair
    else:
      return None, None

  def get_full_folder_path(self):
    """Returns path of directory with all the extracted files."""
    file_type = "eval_data"
    folder_name, _, _ = self.location_info[file_type]
    folder = os.path.join(self.temp_directory, folder_name)
    return folder

  def list_files_for_lang(self, lang):
    """Lists the full paths of all the files for a given language pair."""
    year = self.year
    source_file = "DAseg.newstest{}.source.{}".format(str(year), lang)
    reference_file = "DAseg.newstest{}.reference.{}".format(str(year), lang)
    candidate_file = "DAseg.newstest{}.mt-system.{}".format(str(year), lang)
    rating_file = "DAseg.newstest{}.human.{}".format(str(year), lang)
    folder = self.get_full_folder_path()
    return {
        "source": os.path.join(folder, source_file),
        "reference": os.path.join(folder, reference_file),
        "candidate": os.path.join(folder, candidate_file),
        "rating": os.path.join(folder, rating_file)
    }

  def list_lang_pairs(self):
    folder = self.get_full_folder_path()
    file_names = os.listdir(folder)
    file_data = [Importer1516.parse_file_name(f) for f in file_names]
    lang_pairs = [lang_pair for year, lang_pair in file_data \
                            if year and lang_pair]
    return list(set(lang_pairs))

  def generate_records_for_lang(self, lang):
    year = self.year
    input_files = self.list_files_for_lang(lang)

    # pylint: disable=g-backslash-continuation
    with open(input_files["source"], "r", encoding="utf-8") as source_file, \
         open(input_files["reference"], "r", encoding="utf-8") as reference_file, \
         open(input_files["candidate"], "r", encoding="utf-8") as candidate_file, \
         open(input_files["rating"], "r", encoding="utf-8") as rating_file:
      # pylint: enable=g-backslash-continuation
      n_records = 0
      with open(self.target_file, "a+") as dest_file:
        for source, reference, candidate, rating in itertools.zip_longest(
            source_file, reference_file, candidate_file, rating_file):
          example = Importer1516.to_json(year, lang, source, reference,
                                         candidate, rating, n_records + 1, None)
          dest_file.write(example)
          dest_file.write("\n")
          n_records += 1
      logging.info("Processed {} records".format(str(n_records)))
      return n_records


class Importer17(WMTImporter):
  """Importer for year 2017."""

  def __init__(self, *args, **kwargs):
    super(Importer17, self).__init__(*args, **kwargs)
    self.lang_pairs = None

  def get_folder_path(self):
    """Returns path of directory with all the extracted files."""
    return self.temp_directory

  def agg_ratings_path(self):
    return os.path.join(self.temp_directory, "manual-evaluation",
                        "DA-seglevel.csv")

  def segments_path(self, subset="root"):
    """Return the path to the source, reference, candidate, and raw rating segments.

    Args:
      subset: one if "root", "source", "reference", "candidate", or
        "raw_rating".

    Returns:
      Path to the relevant folder.
    """
    assert subset in ["root", "source", "reference", "candidate", "raw_rating"]
    root_dir = os.path.join(self.temp_directory, "extracted_wmt_package")
    if subset == "root":
      return root_dir

    root_dir = os.path.join(root_dir, "wmt17-metrics-task-no-hybrids")
    if subset == "source":
      return os.path.join(root_dir, "wmt17-submitted-data", "txt", "sources")
    elif subset == "reference":
      return os.path.join(root_dir, "wmt17-submitted-data", "txt", "references")
    elif subset == "candidate":
      return os.path.join(root_dir, "wmt17-submitted-data", "txt",
                          "system-outputs", "newstest2017")
    elif subset == "raw_rating":
      return os.path.join(root_dir, "newstest2017-segment-level-human")

  def fetch_files(self):
    """Downloads the WMT eval files."""
    # Downloads the main archive.
    super(Importer17, self).fetch_files()

    # Unpacks the segments.
    package_path = self.get_folder_path()
    segments_archive = os.path.join(package_path, "input",
                                    "wmt17-metrics-task-no-hybrids.tgz")
    with (tarfile.open(segments_archive, "r:gz")) as tar:
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner) 
          
      
      safe_extract(tar, path=self.segments_path())
    logging.info("Unpacked the segments to {}.".format(self.segments_path()))

    # Gets the language pair names.
    ratings_path = self.agg_ratings_path()
    lang_pairs = set()
    with open(ratings_path, "r") as ratings_file:
      for l in itertools.islice(ratings_file, 1, None):
        lang = l.split(" ")[0]
        assert re.match("[a-z][a-z]-[a-z][a-z]", lang)
        lang_pairs.add(lang)
    self.lang_pairs = list(lang_pairs)
    logging.info("Done")

  def list_lang_pairs(self):
    """List all language pairs included in the WMT files for the target year."""
    assert self.lang_pairs
    return self.lang_pairs

  def get_ref_segments(self, lang):
    """Fetches source and reference translation segments for language pair."""
    src_subfolder = self.segments_path("source")
    ref_subfolder = self.segments_path("reference")
    src_lang, tgt_lang = separate_lang_pair(lang)
    src_file = "newstest2017-{src}{tgt}-src.{lang}".format(
        src=src_lang, tgt=tgt_lang, lang=src_lang)
    ref_file = "newstest2017-{src}{tgt}-ref.{lang}".format(
        src=src_lang, tgt=tgt_lang, lang=tgt_lang)
    src_path = os.path.join(src_subfolder, src_file)
    ref_path = os.path.join(ref_subfolder, ref_file)

    logging.info("Reading data from files {} and {}".format(src_path, ref_path))
    with open(src_path, "r", encoding="utf-8") as f_src:
      src_segments = f_src.readlines()
    with open(ref_path, "r", encoding="utf-8") as f_ref:
      ref_segments = f_ref.readlines()
    src_segments = [postprocess_segment(s) for s in src_segments]
    ref_segments = [postprocess_segment(s) for s in ref_segments]
    logging.info("Read {} source and {} reference segments.".format(
        len(src_segments), len(ref_segments)))
    return src_segments, ref_segments

  @staticmethod
  def parse_submission_file_name(fname):
    """Extracts system names from the name of submission files."""
    wmt_pattern = re.compile(
        r"^newstest2017\.([a-zA-Z0-9\-\.]+\.[0-9]+)\.[a-z]{2}-[a-z]{2}")
    match = re.match(wmt_pattern, fname)
    if match:
      return match.group(1)
    else:
      return None

  def get_sys_segments(self, lang):
    """Builds a dictionary with the generated segments for each system."""
    # Gets all submission file paths.
    root_folder = self.segments_path("candidate")
    folder = os.path.join(root_folder, lang)
    all_files = os.listdir(folder)
    logging.info("Reading submission files from {}".format(folder))

    # Extracts the generated segments for each submission.
    sys_segments = {}
    for sys_file_name in all_files:
      sys_name = Importer17.parse_submission_file_name(sys_file_name)
      assert sys_name
      sys_path = os.path.join(folder, sys_file_name)
      with open(sys_path, "r", encoding="utf-8") as f_sys:
        sys_lines = f_sys.readlines()
        sys_lines = [postprocess_segment(s) for s in sys_lines]
        sys_segments[sys_name] = sys_lines

    logging.info("Read submissions from {} systems".format(
        len(sys_segments.keys())))
    return sys_segments

  def get_raw_rating_scores(self, lang):
    """Builds a dictionary with the rating score for each segment."""
    # Gets the raw ratings file path.
    folder_name, _, _ = self.location_info["eval_data"]
    raw_rating_path = os.path.join(self.temp_directory, folder_name,
                                   "anon-proc-hits-seg-{}".format(lang[-2:]),
                                   "analysis", "ad-seg-scores.csv.gz")
    logging.info("Reading raw ratings from {}".format(raw_rating_path))

    # Extracts the raw rating segments.
    with gzip.open(raw_rating_path, "rt") as f_raw_ratings:
      raw_rating_lines = f_raw_ratings.readlines()
    # Each column in ratings file is separated by spaces.
    raw_rating_lines = [
        postprocess_segment(s).split() for s in raw_rating_lines
    ]
    # Filter out ratings for other language pairs.
    check_lang = lambda x: "-".join([x[0], x[1]]) == lang
    raw_rating_lines = list(filter(check_lang, raw_rating_lines))
    # Create tuple from seg_id (index 5) to raw_rating (index 7).
    raw_ratings = collections.defaultdict(list)
    for x in raw_rating_lines:
      raw_ratings[int(x[5])].append(float(x[7]))

    # If there are multiple ratings, the final rating is averaged.
    for key, value in raw_ratings.items():
      raw_ratings[key] = np.mean(value)
    return raw_ratings

  def parse_rating(self, line):
    fields = line.split()
    lang = fields[0]
    sys_names = fields[2].split("+")
    seg_id = int(fields[3])
    z_score = float(fields[4])
    for sys_name in sys_names:
      yield lang, sys_name, seg_id, z_score

  def generate_records_for_lang(self, lang):
    """Consolidates all the files for a given language pair and year."""
    # Loads source, reference, system segments, and raw ratings.
    src_segments, ref_segments = self.get_ref_segments(lang)
    sys_segments = self.get_sys_segments(lang)
    raw_rating_scores = self.get_raw_rating_scores(lang)

    # Streams the rating file and performs the join on-the-fly.
    ratings_file_path = self.agg_ratings_path()
    logging.info("Reading file {}".format(ratings_file_path))
    n_records = 0
    with open(ratings_file_path, "r", encoding="utf-8") as f_ratings:
      with open(self.target_file, "a+") as dest_file:
        for line in itertools.islice(f_ratings, 1, None):
          for parsed_line in self.parse_rating(line):
            line_lang, sys_name, seg_id, z_score = parsed_line
            if line_lang != lang:
              continue
            # The "-1" is necessary because seg_id starts counting at 1.
            src_segment = src_segments[seg_id - 1]
            ref_segment = ref_segments[seg_id - 1]
            sys_segment = sys_segments[sys_name][seg_id - 1]
            # Directly use seg_id because seg_id is key here, not an index.
            raw_rating_score = raw_rating_scores[seg_id]
            example = Importer18.to_json(self.year, lang, src_segment,
                                         ref_segment, sys_segment,
                                         raw_rating_score, z_score, seg_id,
                                         sys_name)
            dest_file.write(example)
            dest_file.write("\n")
            n_records += 1
    logging.info(
        "Done reading ratings file. {} records written.".format(n_records))
    return n_records


class Importer18(WMTImporter):
  """Importer for year 2018."""

  def parse_submission_file_name(self, fname):
    """Extracts system names from the name of submission files."""
    wmt_pattern = re.compile(
        r"^newstest2018\.([a-zA-Z0-9\-\.]+\.[0-9]+)\.[a-z]{2}-[a-z]{2}")
    match = re.match(wmt_pattern, fname)
    if match:
      return match.group(1)
    else:
      return None

  def parse_eval_file_name(self, fname):
    """Extracts language pairs from the names of human rating files."""
    wmt_pattern = re.compile(r"^ad-seg-scores-([a-z]{2}-[a-z]{2})\.csv")
    match = re.match(wmt_pattern, fname)
    if match:
      return match.group(1)
    else:
      return None

  def list_lang_pairs(self):
    """List all language pairs included in the WMT files for 2018."""
    folder_name, _, _ = self.location_info["eval_data"]
    subfolder = "analysis"
    folder = os.path.join(self.temp_directory, folder_name, subfolder)
    all_files = os.listdir(folder)
    cand_lang_pairs = [self.parse_eval_file_name(fname) for fname in all_files]
    # We need to remove None values in cand_lang_pair:
    lang_pairs = [lang_pair for lang_pair in cand_lang_pairs if lang_pair]
    return list(set(lang_pairs))

  def get_ref_segments(self, lang):
    """Fetches source and reference translation segments for language pair."""
    folder, _, _ = self.location_info["submissions"]
    src_subfolder = os.path.join("sources")
    ref_subfolder = os.path.join("references")
    src_lang, tgt_lang = separate_lang_pair(lang)
    src_file = "newstest2018-{src}{tgt}-src.{lang}".format(
        src=src_lang, tgt=tgt_lang, lang=src_lang)
    ref_file = "newstest2018-{src}{tgt}-ref.{lang}".format(
        src=src_lang, tgt=tgt_lang, lang=tgt_lang)
    src_path = os.path.join(self.temp_directory, folder, src_subfolder,
                            src_file)
    ref_path = os.path.join(self.temp_directory, folder, ref_subfolder,
                            ref_file)

    logging.info("Reading data from files {} and {}".format(src_path, ref_path))
    with open(src_path, "r", encoding="utf-8") as f_src:
      src_segments = f_src.readlines()
    with open(ref_path, "r", encoding="utf-8") as f_ref:
      ref_segments = f_ref.readlines()

    src_segments = [postprocess_segment(s) for s in src_segments]
    ref_segments = [postprocess_segment(s) for s in ref_segments]

    return src_segments, ref_segments

  def get_sys_segments(self, lang):
    """Builds a dictionary with the generated segments for each system."""
    # Gets all submission file paths.
    folder_name, _, _ = self.location_info["submissions"]
    subfolder = os.path.join("system-outputs", "newstest2018")
    folder = os.path.join(self.temp_directory, folder_name, subfolder, lang)
    all_files = os.listdir(folder)
    logging.info("Reading submission files from {}".format(folder))

    # Extracts the generated segments for each submission.
    sys_segments = {}
    for sys_file_name in all_files:
      sys_name = self.parse_submission_file_name(sys_file_name)
      assert sys_name
      sys_path = os.path.join(folder, sys_file_name)
      with open(sys_path, "r", encoding="utf-8") as f_sys:
        sys_lines = f_sys.readlines()
        sys_lines = [postprocess_segment(s) for s in sys_lines]
        sys_segments[sys_name] = sys_lines

    return sys_segments

  def get_ratings_path(self, lang):
    folder, _, _ = self.location_info["eval_data"]
    subfolder = "analysis"
    file_name = "ad-seg-scores-{}.csv".format(lang)
    return os.path.join(self.temp_directory, folder, subfolder, file_name)

  def parse_rating(self, rating_line):
    rating_tuple = tuple(rating_line.split(" "))
    # I have a feeling that the last field is the number of ratings
    # but I'm not 100% sure .
    sys_name, seg_id, raw_score, z_score, n_ratings = rating_tuple
    seg_id = int(seg_id)
    raw_score = float(raw_score)
    z_score = float(z_score)
    n_ratings = int(n_ratings)
    return sys_name, seg_id, raw_score, z_score, n_ratings

  @staticmethod
  def to_json(year,
              lang,
              src_segment,
              ref_segment,
              sys_segment,
              raw_score,
              z_score,
              seg_id,
              sys_name,
              n_ratings=0):
    """Converts record to JSON."""
    json_dict = {
        "year": year,
        "lang": lang,
        "source": src_segment,
        "reference": ref_segment,
        "candidate": sys_segment,
        "raw_rating": raw_score,
        "rating": z_score,
        "segment_id": seg_id,
        "system": sys_name,
        "n_ratings": n_ratings
    }
    return json.dumps(json_dict)

  def generate_records_for_lang(self, lang):
    """Consolidates all the files for a given language pair and year."""

    # Loads source, reference and system segments.
    src_segments, ref_segments = self.get_ref_segments(lang)
    sys_segments = self.get_sys_segments(lang)

    # Streams the rating file and performs the join on-the-fly.
    ratings_file_path = self.get_ratings_path(lang)
    logging.info("Reading file {}".format(ratings_file_path))
    n_records = 0
    with open(ratings_file_path, "r", encoding="utf-8") as f_ratings:
      with open(self.target_file, "a+") as dest_file:
        for line in itertools.islice(f_ratings, 1, None):
          line = line.rstrip()
          parsed_tuple = self.parse_rating(line)
          sys_name, seg_id, raw_score, z_score, n_ratings = parsed_tuple

          # Those weird rules come from the WMT 2019 DA2RR script.
          # Name of the script: seglevel-ken-rr.py, in Metrics results package.
          if sys_name == "UAlacant_-_NM":
            sys_name = "UAlacant_-_NMT+RBMT.6722"
          if sys_name == "HUMAN":
            continue
          if sys_name == "RBMT.6722":
            continue

          # The following rules were added by me to unblock WMT2019:
          if sys_name == "Helsinki-NLP.6889":
            sys_name = "Helsinki_NLP.6889"
          if sys_name == "Facebook-FAIR.6937":
            sys_name = "Facebook_FAIR.6937"
          if sys_name == "Facebook-FAIR.6937":
            sys_name = "Facebook_FAIR.6937"
          if sys_name == "DBMS-KU-KKEN.6726":
            sys_name = "DBMS-KU_KKEN.6726"
          if sys_name == "Ju-Saarland.6525":
            sys_name = "Ju_Saarland.6525"
          if sys_name == "aylien-mt-gu-en-multilingual.6826":
            sys_name = "aylien_mt_gu-en_multilingual.6826"
          if sys_name == "rug-kken-morfessor.6677":
            sys_name = "rug_kken_morfessor.6677"
          if sys_name == "talp-upc-2019-kken.6657":
            sys_name = "talp_upc_2019_kken.6657"
          if sys_name == "Frank-s-MT.6127":
            sys_name = "Frank_s_MT.6127"

          if lang == "de-cs" and sys_name == "Unsupervised.6935":
            sys_name = "Unsupervised.de-cs.6935"
          if lang == "de-cs" and sys_name == "Unsupervised.6929":
            sys_name = "Unsupervised.de-cs.6929"

          # The "-1" is necessary because seg_id starts counting at 1.
          src_segment = src_segments[seg_id - 1]
          ref_segment = ref_segments[seg_id - 1]
          sys_segment = sys_segments[sys_name][seg_id - 1]
          if not src_segment or not sys_segment:
            logging.info("* Missing value!")
            logging.info("* System:" + sys_name)
            logging.info("* Segment:" + str(seg_id))
            logging.info("* Source segment:" + src_segment)
            logging.info("* Sys segment:" + sys_segment)
            logging.info("* Parsed line:" + line)
            logging.info("* Lang:" + lang)
          example = Importer18.to_json(self.year, lang, src_segment,
                                       ref_segment, sys_segment, raw_score,
                                       z_score, seg_id, sys_name, n_ratings)
          dest_file.write(example)
          dest_file.write("\n")
          n_records += 1
    logging.info("Done reading ratings file")
    return n_records


class Importer19(Importer18):
  """Importer for WMT19 Metrics challenge."""

  def parse_rating(self, rating_line):
    rating_tuple = tuple(rating_line.split(" "))
    # I have a feeling that the last field is the number of ratings
    # but I'm not 100% sure.
    sys_name, seg_id, raw_score, z_score, n_ratings = rating_tuple

    # For some reason, all the systems for pair zh-en have an extra suffix.
    if sys_name.endswith("zh-en"):
      sys_name = sys_name[:-6]

    seg_id = int(seg_id)
    raw_score = float(raw_score)
    z_score = float(z_score)
    n_ratings = int(n_ratings)
    return sys_name, seg_id, raw_score, z_score, n_ratings

  def parse_submission_file_name(self, fname):
    """Extracts system names from the name of submission files."""

    # I added those rules to unblock the pipeline.
    if fname == "newstest2019.Unsupervised.de-cs.6929.de-cs":
      return "Unsupervised.de-cs.6929"
    elif fname == "newstest2019.Unsupervised.de-cs.6935.de-cs":
      return "Unsupervised.de-cs.6935"

    wmt_pattern = re.compile(
        r"^newstest2019\.([a-zA-Z0-9\-\.\_\+]+\.[0-9]+)\.[a-z]{2}-[a-z]{2}")
    match = re.match(wmt_pattern, fname)
    if match:
      return match.group(1)
    else:
      return None

  def list_lang_pairs(self):
    """List all language pairs included in the WMT files for 2019."""
    folder_name, _, _ = self.location_info["eval_data"]
    folder = os.path.join(self.temp_directory, folder_name, "*", "analysis",
                          "ad-seg-scores-*.csv")
    all_full_paths = glob.glob(folder)
    all_files = [os.path.basename(f) for f in all_full_paths]
    cand_lang_pairs = [self.parse_eval_file_name(fname) for fname in all_files]
    # We need to remove None values in cand_lang_pair:
    lang_pairs = [lang_pair for lang_pair in cand_lang_pairs if lang_pair]
    return list(set(lang_pairs))

  def get_ratings_path(self, lang):
    folder, _, _ = self.location_info["eval_data"]

    # The pair zh-en has two versions in the WMT 2019 human eval folder.
    if lang == "zh-en":
      path = os.path.join(self.temp_directory, folder,
                          "turkle-sntlevel-humaneval-newstest2019", "analysis",
                          "ad-seg-scores-zh-en.csv")
      return path

    file_name = "ad-seg-scores-{}.csv".format(lang)
    folder = os.path.join(self.temp_directory, folder, "*", "analysis",
                          "ad-seg-scores-*.csv")
    all_files = glob.glob(folder)
    for cand_file in all_files:
      if cand_file.endswith(file_name):
        return cand_file
    raise ValueError("Can't find ratings for lang {}".format(lang))

  def get_ref_segments(self, lang):
    """Fetches source and reference translation segments for language pair."""
    folder, _, _ = self.location_info["submissions"]
    src_subfolder = os.path.join("txt", "sources")
    ref_subfolder = os.path.join("txt", "references")
    src_lang, tgt_lang = separate_lang_pair(lang)
    src_file = "newstest2019-{src}{tgt}-src.{lang}".format(
        src=src_lang, tgt=tgt_lang, lang=src_lang)
    ref_file = "newstest2019-{src}{tgt}-ref.{lang}".format(
        src=src_lang, tgt=tgt_lang, lang=tgt_lang)
    src_path = os.path.join(self.temp_directory, folder, src_subfolder,
                            src_file)
    ref_path = os.path.join(self.temp_directory, folder, ref_subfolder,
                            ref_file)

    logging.info("Reading data from files {} and {}".format(src_path, ref_path))
    with open(src_path, "r", encoding="utf-8") as f_src:
      src_segments = f_src.readlines()
    with open(ref_path, "r", encoding="utf-8") as f_ref:
      ref_segments = f_ref.readlines()

    src_segments = [postprocess_segment(s) for s in src_segments]
    ref_segments = [postprocess_segment(s) for s in ref_segments]

    return src_segments, ref_segments

  def get_sys_segments(self, lang):
    """Builds a dictionary with the generated segments for each system."""
    # Gets all submission file paths.
    folder_name, _, _ = self.location_info["submissions"]
    subfolder = os.path.join("txt", "system-outputs", "newstest2019")
    folder = os.path.join(self.temp_directory, folder_name, subfolder, lang)
    all_files = os.listdir(folder)
    logging.info("Reading submission files from {}".format(folder))

    # Extracts the generated segments for each submission.
    sys_segments = {}
    for sys_file_name in all_files:
      sys_name = self.parse_submission_file_name(sys_file_name)
      assert sys_name
      sys_path = os.path.join(folder, sys_file_name)
      with open(sys_path, "r", encoding="utf-8") as f_sys:
        sys_lines = f_sys.readlines()
        sys_lines = [postprocess_segment(s) for s in sys_lines]
        sys_segments[sys_name] = sys_lines

    return sys_segments
