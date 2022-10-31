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
"""BLEURT's Tensorflow ops."""

from bleurt import checkpoint as checkpoint_lib
from bleurt.lib import optimization
import numpy as np
from scipy import stats
import tensorflow.compat.v1 as tf

from tensorflow.compat.v1 import estimator as tf_estimator
from tf_slim import metrics
from bleurt.lib import modeling

flags = tf.flags
logging = tf.logging
FLAGS = flags.FLAGS

# BLEURT flags.
flags.DEFINE_string("bleurt_checkpoint_name", "bert_custom",
                    "Name of the BLEURT export to be created.")

flags.DEFINE_string("init_bleurt_checkpoint", None,
                    "Existing BLEURT export to be fine-tuned.")

# BERT flags.
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", None,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("dynamic_seq_length", True,
                  "Exports model with dymaic sequence length.")

# Flags to control training setup.
flags.DEFINE_enum("export_metric", "kendalltau", ["correlation", "kendalltau"],
                  "Metric to chose the best model in export functions.")

flags.DEFINE_integer("shuffle_buffer_size", 500,
                     "Size of buffer used to shuffle the examples.")

# Flags to contol optimization.
flags.DEFINE_enum("optimizer", "adam", ["adam", "sgd", "adagrad"],
                  "Which optimizer to use.")

flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool(
    "use_ranking_loss", False,
    "Whether to use a ranking loss instead of regression (l2 loss).")

# BLEURT model flags.
flags.DEFINE_integer("n_hidden_layers", 0,
                     "Number of fully connected/RNN layers before prediction.")

flags.DEFINE_integer("hidden_layers_width", 128, "Width of hidden layers.")

flags.DEFINE_float("dropout_rate", 0,
                   "Probability of dropout over BERT embedding.")


flags.DEFINE_float("random_seed", 55555, "Random seed for TensorFlow.")


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, use_one_hot_embeddings, n_hidden_layers,
                 hidden_layers_width, dropout_rate):
  """Creates a regression model, loosely adapted from language/bert.

  Args:
    bert_config: `BertConfig` instance.
    is_training:  bool. true for training model, false for eval model.
    input_ids: int32 Tensor of shape [batch_size, seq_length].
    input_mask: int32 Tensor of shape [batch_size, seq_length].
    segment_ids: int32 Tensor of shape [batch_size, seq_length].
    labels: float32 Tensor of shape [batch_size].
    use_one_hot_embeddings:  Whether to use one-hot word embeddings or
      tf.embedding_lookup() for the word embeddings.
    n_hidden_layers: number of FC layers before prediction.
    hidden_layers_width: width of FC layers.
    dropout_rate: probability of dropout over BERT embedding.

  Returns:
    loss: <float32>[]
    per_example_loss: <float32>[batch_size]
    pred: <float32>[batch_size]
  """
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # <float>[batch_size, hidden_size]
  output_layer = model.get_pooled_output()
  bert_embed_size = output_layer.shape[-1]
  logging.info("BERT embedding width: {}".format(str(bert_embed_size)))
  if is_training and dropout_rate > 0:
    # Implements dropout on top of BERT's pooled output.
    # <float32>[batch_size, hidden_size]
    output_layer = tf.nn.dropout(output_layer, rate=dropout_rate)

  # Hidden layers
  for i in range(n_hidden_layers):
    # <float32>[batch_size, hidden_layers_width]
    logging.info("Adding hidden layer {}".format(i + 1))
    output_layer = tf.layers.dense(
        output_layer, hidden_layers_width, activation=tf.nn.relu)

  logging.info("Building linear output...")
  # <float32>[batch_size,1]
  predictions = tf.layers.dense(
      output_layer, 1, bias_initializer=tf.constant_initializer(0.15))
  # <float32>[batch_size]
  predictions = tf.squeeze(predictions, 1)
  # <float32>[batch_size]
  if FLAGS.use_ranking_loss:
    per_example_loss = ranking_loss(predictions, labels)
  else:
    per_example_loss = tf.pow(predictions - labels, 2)
  # <float32> []
  loss = tf.reduce_mean(per_example_loss, axis=-1)

  return (loss, per_example_loss, predictions)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, n_hidden_layers,
                     hidden_layers_width, dropout_rate):
  """Returns `model_fn` closure."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator/TPUEstimator."""

    logging.info("*** Building Regression BERT Model ***")
    tf.set_random_seed(FLAGS.random_seed)

    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s", name, features[name].shape)

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    if mode != tf_estimator.ModeKeys.PREDICT:
      scores = features["score"]
    else:
      scores = tf.zeros(tf.shape(input_ids)[0])

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)
    total_loss, per_example_loss, pred = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, scores,
        use_one_hot_embeddings, n_hidden_layers, hidden_layers_width,
        dropout_rate)

    output_spec = None
    if mode == tf_estimator.ModeKeys.TRAIN:

      # Loads pretrained model
      logging.info("**** Initializing from {} ****".format(init_checkpoint))
      tvars = tf.trainable_variables()
      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:
          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()
          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                     init_string)

      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      if use_tpu:
        output_spec = tf_estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)

      else:
        output_spec = tf_estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op)

    elif mode == tf_estimator.ModeKeys.EVAL:

      if use_tpu:
        eval_metrics = (metric_fn, [per_example_loss, pred, scores])
        output_spec = tf_estimator.TPUEstimatorSpec(
            mode=mode, loss=total_loss, eval_metric=eval_metrics)
      else:
        output_spec = tf_estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_fn(per_example_loss, pred, scores))

    elif mode == tf_estimator.ModeKeys.PREDICT:
      output_spec = tf_estimator.EstimatorSpec(
          mode=mode, predictions={"predictions": pred})

    return output_spec

  return model_fn


def ranking_loss(predictions, labels):
  """Ranking loss as described in https://arxiv.org/pdf/2009.01325.pdf."""
  # We found that clipping the predictions during training helps, since the
  # ranking loss itself does not constrain the predictions.
  # TODO(tsellam): Understand why clipping helps.
  predictions = tf.clip_by_value(predictions, 0.0, 1.0)
  # Gets pairs of predictions and pairs of labels in the same order.
  ii, jj = tf.meshgrid(
      tf.range(0,
               tf.shape(predictions)[0]),
      tf.range(0,
               tf.shape(predictions)[0]),
      indexing="ij")
  indices = tf.stack([tf.reshape(ii, [-1]), tf.reshape(jj, [-1])], axis=1)
  indices = tf.boolean_mask(indices, indices[:, 0] < indices[:, 1])
  prediction_pairs = tf.gather(predictions, indices, axis=0)
  label_pairs = tf.gather(labels, indices, axis=0)

  # For each pair, the loss is - log(sigmoid(s_i - s_j)) where i is the better
  # translation according to the labels and s_k denotes the predicted score
  # for translation k.
  score_diffs = (prediction_pairs[:, 0] - prediction_pairs[:, 1]) * (
      tf.cast(label_pairs[:, 0] > label_pairs[:, 1], tf.float32) * 2 - 1)
  per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(score_diffs), logits=score_diffs)
  return per_example_loss


# TF ops to compute the metrics.
def concat_tensors(predictions, ratings, sources=None):
  """Concatenates batches of ratings and predictions."""
  concat_predictions_value, concat_predictions_update = (
      metrics.streaming_concat(predictions))
  concat_labels_value, concat_labels_update = (
      metrics.streaming_concat(ratings))
  if sources is None:
    return (concat_predictions_value, concat_labels_value,
            tf.group(concat_predictions_update, concat_labels_update))

  concat_sources_value, concat_sources_update = (
      metrics.streaming_concat(sources))
  return (concat_predictions_value, concat_labels_value, concat_sources_value,
          tf.group(concat_predictions_update, concat_labels_update,
                   concat_sources_update))


def kendall_tau_metric(predictions, ratings, weights=None):
  """Builds the computation graph for Kendall Tau metric."""

  def _kendall_tau(x, y):
    tau = stats.kendalltau(x, y)[0]
    return np.array(tau).astype(np.float32)

  if weights is not None:
    predictions = tf.boolean_mask(predictions, weights)
    ratings = tf.boolean_mask(ratings, weights)

  with tf.variable_scope("kendall_tau"):
    concat_predictions_value, concat_labels_value, update_op = (
        concat_tensors(predictions, ratings))
    metric_value = tf.reshape(
        tf.numpy_function(_kendall_tau,
                          [concat_predictions_value, concat_labels_value],
                          tf.float32),
        shape=[])

    return metric_value, update_op


def metric_fn(per_example_loss, pred, ratings):
  """Metrics for BLEURT experiments."""
  # Mean of predictions
  mean_pred = tf.metrics.mean(values=pred)
  # Standard deviation of predictions
  mean = tf.reduce_mean(pred)
  diffs = tf.sqrt(tf.pow(pred - mean, 2))
  pred_sd = tf.metrics.mean(values=diffs)
  # Average squared error
  mean_loss = tf.metrics.mean(values=per_example_loss)
  # Average absolute error
  squared_diff = tf.pow(pred - ratings, 2)
  per_example_err = tf.sqrt(squared_diff)
  mean_err = tf.metrics.mean(per_example_err)
  # Pearson correlation
  correlation = metrics.streaming_pearson_correlation(pred, ratings)
  # Kendall Tau
  kendalltau = kendall_tau_metric(pred, ratings)
  output = {
      "eval_loss": mean_loss,
      "eval_mean_err": mean_err,
      "eval_mean_pred": mean_pred,
      "eval_pred_sd": pred_sd,
      "correlation": correlation,
      "kendalltau": kendalltau,
  }

  return output


def input_fn_builder(tfrecord_file,
                     seq_length,
                     is_training,
                     batch_size,
                     drop_remainder=True):
  """Creates an `input_fn` closure to be passed to Estimator."""
  logging.info(
      "Creating input fun with batch_size: {} and drop remainder: {}".format(
          str(batch_size), str(drop_remainder)))
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "score": tf.FixedLenFeature([], tf.float32)
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t
    return example

  def input_fn(params):  # pylint: disable=unused-argument
    """Acutal data generator."""
    tfrecord_file_expanded = tf.io.gfile.glob(tfrecord_file)
    n_files = len(tfrecord_file_expanded)
    if n_files > 1:
      logging.info("Found {} files matching {}".format(
          str(n_files), tfrecord_file))

    d = tf.data.TFRecordDataset(tfrecord_file_expanded)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    d = d.map(lambda record: _decode_record(record, name_to_features))
    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def _model_comparator(best_eval_result, current_eval_result):
  metric = FLAGS.export_metric
  return best_eval_result[metric] <= current_eval_result[metric]


def _serving_input_fn_builder(seq_length):
  """Input function for exported models."""
  # We had to use `tf.zeros` instead of the usual
  # `tf.placeholder(tf.int64, shape=[None, seq_length])` to be compatible with
  # TF2's eager mode, which deprecates all calls to `tf.placeholder`.
  if tf.executing_eagerly():
    assert not FLAGS.dynamic_seq_length, (
        "Training with `dynamic_seq_length is not supported in Eager mode.")
    name_to_features = {
        "input_ids": tf.zeros(dtype=tf.int64, shape=[0, seq_length]),
        "input_mask": tf.zeros(dtype=tf.int64, shape=[0, seq_length]),
        "segment_ids": tf.zeros(dtype=tf.int64, shape=[0, seq_length])
    }
  else:
    if FLAGS.dynamic_seq_length:
      logging.info("Exporting a model with dynamic sequence length.")
      name_to_features = {
          "input_ids": tf.placeholder(tf.int64, shape=[None, None]),
          "input_mask": tf.placeholder(tf.int64, shape=[None, None]),
          "segment_ids": tf.placeholder(tf.int64, shape=[None, None])
      }
    else:
      name_to_features = {
          "input_ids": tf.placeholder(tf.int64, shape=[None, seq_length]),
          "input_mask": tf.placeholder(tf.int64, shape=[None, seq_length]),
          "segment_ids": tf.placeholder(tf.int64, shape=[None, seq_length])
      }
  return tf_estimator.export.build_raw_serving_input_receiver_fn(
      name_to_features)


def run_finetuning(train_tfrecord,
                   dev_tfrecord,
                   train_eval_fun=None,
                   use_tpu=False,
                   multi_eval_names=None,
                   additional_train_params=None):
  """Main function to train and eval BLEURT."""

  logging.info("Initializing BLEURT training pipeline.")

  bleurt_params = checkpoint_lib.get_bleurt_params_from_flags_or_ckpt()
  max_seq_length = bleurt_params["max_seq_length"]
  bert_config_file = bleurt_params["bert_config_file"]
  init_checkpoint = bleurt_params["init_checkpoint"]

  logging.info("Creating input data pipeline.")
  logging.info("Train batch size: {}, eval batch size: {}".format(
      str(FLAGS.batch_size), str(FLAGS.eval_batch_size)))
  logging.info("Train data: {}".format(train_tfrecord))

  train_input_fn = input_fn_builder(
      train_tfrecord,
      seq_length=max_seq_length,
      is_training=True,
      batch_size=FLAGS.batch_size,
      drop_remainder=use_tpu)

  additional_eval_specs = None
  if isinstance(dev_tfrecord, str):
    logging.info("Validation data: {}".format(dev_tfrecord))
    dev_input_fn = input_fn_builder(
        dev_tfrecord,
        seq_length=max_seq_length,
        is_training=False,
        batch_size=FLAGS.eval_batch_size,
        drop_remainder=use_tpu)
  elif isinstance(dev_tfrecord, list) and dev_tfrecord:
    logging.info("Validation data: {}".format(",".join(dev_tfrecord)))
    dev_input_fn = input_fn_builder(
        dev_tfrecord[0],
        seq_length=max_seq_length,
        is_training=False,
        batch_size=FLAGS.eval_batch_size,
        drop_remainder=use_tpu)
    additional_eval_specs = []
    for i in range(1, len(dev_tfrecord)):
      additional_dev_input_fn = input_fn_builder(
          dev_tfrecord[i],
          seq_length=max_seq_length,
          is_training=False,
          batch_size=FLAGS.eval_batch_size,
          drop_remainder=use_tpu)
      eval_name = multi_eval_names[i] if multi_eval_names and len(
          multi_eval_names) > i else "eval_%s" % i
      additional_eval_specs.append(
          tf_estimator.EvalSpec(
              name=eval_name,
              input_fn=additional_dev_input_fn,
              steps=FLAGS.num_eval_steps))
      logging.info("len(additional_eval_specs): ", len(additional_eval_specs))
  else:
    raise ValueError(
        "dev_tfrecord can only be a string or list: {}".format(dev_tfrecord))

  logging.info("Creating model.")
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)
  num_train_steps = FLAGS.num_train_steps
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=use_tpu,
      use_one_hot_embeddings=use_tpu,
      n_hidden_layers=FLAGS.n_hidden_layers,
      hidden_layers_width=FLAGS.hidden_layers_width,
      dropout_rate=FLAGS.dropout_rate)

  logging.info("Creating TF Estimator.")
  exporters = [
      tf_estimator.BestExporter(
          "bleurt_best",
          serving_input_receiver_fn=_serving_input_fn_builder(max_seq_length),
          event_file_pattern="eval_default/*.tfevents.*",
          compare_fn=_model_comparator,
          exports_to_keep=3)
  ]
  tf.enable_resource_variables()

  logging.info("*** Entering the Training / Eval phase ***")
  if not additional_train_params:
    additional_train_params = {}
  train_eval_fun(
      model_fn=model_fn,
      train_input_fn=train_input_fn,
      eval_input_fn=dev_input_fn,
      additional_eval_specs=additional_eval_specs,
      exporters=exporters,
      **additional_train_params)
