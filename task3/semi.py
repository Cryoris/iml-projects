# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
import numpy as np

from official.utils.arg_parsers import parsers
from official.utils.logs import hooks_helper
from official.utils.misc import model_helpers

_CSV_COLUMNS = ["y"] + ["x{}".format(i) for i in range(1,100+1)]
#_CSV_COLUMNS = ["x{}".format(i) for i in range(1,15+1)]
#_CSV_COLUMNS = ['x1', 'x3', 'x5', 'x7', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']
#_CSV_COLUMN_DEFAULTS = [[0], [0], [0], [''], [''], [''],
#                        [0], [0], [0], [''], ['']]

_CSV_COLUMN_DEFAULTS = [[0]] + [[0.]]*100

"""
_NUM_EXAMPLES = {
    'train': 45324,
    'validation': 8137,
}
"""
_NUM_EXAMPLES = {
    'train': 40000,
    'validation': 5324,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  x = []
  for i in range(100):
      x += [tf.feature_column.numeric_column('x{}'.format(i+1))]

  # Wide columns and deep columns.
  base_columns = []

  """
  crossed_columns = [
      tf.feature_column.crossed_column(
          ['x4', 'x7'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [age_buckets, 'x4', 'x7'], hash_bucket_size=1000),
  ]
  """

  #wide_columns = base_columns + crossed_columns
  wide_columns = base_columns

  deep_columns = [*x,]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type, hidden_units):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config,
        n_classes=5)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config,
        n_classes=5)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        n_classes=5)


def input_fn(data_file, num_epochs, shuffle, batch_size, has_labels=True):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    print(_CSV_COLUMNS)
    print(len(columns))
    features = dict(zip(_CSV_COLUMNS, columns))
    if has_labels:
        labels = features.pop('y')
        return features, labels
    else:
        return features

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset



def main(argv):
  parser = WideDeepArgParser()
  flags = parser.parse_args(args=argv[1:])

  hidden_units = [int(u) for u in flags.hidden_units.strip("[]").split(",")]

  print("STATUS --" * 10)
  print(hidden_units)
  print(flags.model_dir)
  print(flags.model_type)
  print(flags.data_dir)
  print(flags.epochs_between_evals)
  print(flags.batch_size)
  print(flags.train_epochs)

  # Clean up the model directory if present
  shutil.rmtree(flags.model_dir, ignore_errors=True)
  model = build_estimator(flags.model_dir, flags.model_type, hidden_units)

  #train_file = os.path.join(flags.data_dir, 'train_minus.csv')
  train_file = os.path.join(flags.data_dir, 'train.csv')
  test_file = os.path.join(flags.data_dir, 'validate.csv')
  predict_file = os.path.join(flags.data_dir, 'test.csv')

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return input_fn(
        train_file, flags.epochs_between_evals, True, flags.batch_size)

  def eval_input_fn():
    return input_fn(test_file, 1, False, flags.batch_size)

  def predict_input_fn():
    return input_fn(predict_file, 1, False, flags.batch_size, has_labels=False)

  loss_prefix = LOSS_PREFIX.get(flags.model_type, '')
  train_hooks = hooks_helper.get_train_hooks(
      flags.hooks, batch_size=flags.batch_size,
      tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      'loss': loss_prefix + 'head/weighted_loss/Sum'})

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  for n in range(flags.train_epochs // flags.epochs_between_evals):
    model.train(input_fn=train_input_fn, hooks=train_hooks)
    """
    results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * flags.epochs_between_evals)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))

    if model_helpers.past_stop_threshold(
        flags.stop_threshold, results['accuracy']):
      break
    """

  prediction = list(model.predict(input_fn=predict_input_fn))

  # Create array w/ [class, probability of this class]
  # TODO Improve: class prob might be low but still much higher than
  # all other probs, that case would also be okay
  predicted_classes = np.array([[int(p["classes"][0]), max(p["probabilities"])] for p in prediction])
  return prediction, predicted_classes, train_file, test_file


class WideDeepArgParser(argparse.ArgumentParser):
  """Argument parser for running the wide deep model."""

  def __init__(self):
    super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()])
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide_deep',
        choices=['wide', 'deep', 'wide_deep'],
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
        metavar='<MT>')
    self.add_argument(
        '--hidden_units', '-hu', type=str, default="[100, 75, 50, 25]",
        help='List of ints with number of units per layer, default [100, 75, 50, 25]',
        metavar='<HU>')
    self.set_defaults(
        data_dir='/tmp/data',
        model_dir='/tmp/model',
        train_epochs=40,
        epochs_between_evals=2,
        batch_size=40)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  prediction, predicted_classes, train_file, test_file = main(argv=sys.argv)

  # If we're more than 40% sure that the class is correct, label it
  # otherwise don't
  prediction_threshold = 0.4
  # Get indices of which points we should label
  idx = np.where(prediced_classes[:,1] > prediction_threshold)


  import csv
  with open("prediction.csv", "w") as csvfile:
      offset = 45324
      writer = csv.writer(csvfile)
      writer.writerow(["Id","y"])
      for i, predicted_class in enumerate(predicted_classes):
          writer.writerow([offset + i, predicted_class])
