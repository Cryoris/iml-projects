# Copyright 2018 Add schierkherinha
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
# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys
import csv

import tensorflow as tf  # pylint: disable=g-bad-import-order
import numpy as np

from official.utils.arg_parsers import parsers
from official.utils.logs import hooks_helper
from official.utils.misc import model_helpers

# Global definitions
_CSV_COLUMNS = ["y"] + ["x{}".format(i) for i in range(1,128+1)]
#_LABELED_HEADER = str(_CSV_COLUMNS).strip("[]").replace(" ", "").replace("'", "")
#_UNLABELED_HEADER = str(_CSV_COLUMNS[1:]).strip("[]").replace(" ", "").replace("'", "")
#_CSV_COLUMNS = ["x{}".format(i) for i in range(1,15+1)]
#_CSV_COLUMNS = ['x1', 'x3', 'x5', 'x7', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']
#_CSV_COLUMN_DEFAULTS = [[0], [0], [0], [''], [''], [''],
#                        [0], [0], [0], [''], ['']]

_CSV_COLUMN_DEFAULTS = [[0]] + [[0.]]*128
_FMT = "%i" + 128*",%f"

_NUM_EXAMPLES = {'train': 4000}

_NCLASSES = 10
_DATA_DIR = "~/tmp/data"
_MODEL_DIR = "~/tmp/model"
_OFFSET = 30000

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}

class TClassifier:
    def __init__(self, argv, trainfile_labeled, trainfile_unlabeled, testfile,
                 classification_threshold, max_percentage_unlabeled=0.01):
        """
            argv: as in werockthis.py
            trainfile_labeled: labeled%i.csv if labeled training data is in labeled0.csv
            trainfile_unlabeled: unlabeled%i.csv if unlabeled training data is in unlabeled0.csv
            testfile: file with test data
            classification_threshold: minimal probability we need to classify a sample
        """
        self._argv = argv
        self._flabel = trainfile_labeled
        self._funlabel = trainfile_unlabeled
        self._ftest = testfile
        self._thres = classification_threshold
        self._max_p_unlabeled = max_percentage_unlabeled
        self._step = 0

        # Parameters for indoctrine training
        self._max_steps = 10
        self._min_frac = 0.1


    def step(self):
        # Get class predictions and their probabilities
        print("Call to main.")
        print("* flabel:\t\t", self._flabel % self._step)
        print("* funlabel:\t\t", self._funlabel % self._step)
        classprobs, flabel, funlabel = main(self._argv,
                                            self._flabel % self._step,
                                            self._funlabel % self._step)

        print("Done.")
        self._step += 1

        # Get labeled and unlabeled data
        print("Reading labeled and unlabeled data from")
        print("* flabel:\t\t", flabel)
        print("* funlabel:\t\t", funlabel)
        labeled_data = np.genfromtxt(flabel, delimiter=",")
        unlabeled_data = np.genfromtxt(funlabel, delimiter=",")

        # Get indices where probability for predicted class is > thres
        idx_to_label = np.where(classprobs[:,1] > self._thres)[0]
        np.savetxt("cp.%i" % self._step, classprobs)
        print("Set sizes:")
        print("* Labeled:\t\t", labeled_data.shape[0])
        print("* Unlabeled:\t\t", unlabeled_data.shape[0])
        print("* Indices to label:\t", idx_to_label.size)

        # Get fraction of data that's freshly labeled
        frac_labeled = 1.0*idx_to_label.size/unlabeled_data.shape[0]

        # Predict those labels (if they exist)
        if idx_to_label.size > 0:
            predict = unlabeled_data[idx_to_label, :] # Set datapoints
            predict[:,0] = classprobs[idx_to_label, 0] # Set predicted classes

            labeled_data = np.vstack((labeled_data, predict))
            # Remove freshly predicted labels from unlabeled set
            unlabeled_data = np.delete(unlabeled_data, idx_to_label, axis=0)

        # Set global variable for number of training samples
        _NUM_EXAMPLES = {'train': labeled_data.shape[0]}

        # Save new datasets
        print("Write new datasets to")
        print("* flabel:\t", self._flabel % self._step)
        print("* funlabel:\t", self._funlabel % self._step)
        new_flabel = os.path.join(_DATA_DIR, self._flabel % self._step)
        new_funlabel = os.path.join(_DATA_DIR, self._funlabel % self._step)
        np.savetxt(new_flabel, labeled_data, fmt=_FMT)
        np.savetxt(new_funlabel, unlabeled_data, fmt=_FMT)
        print("Done.")

        return frac_labeled


    def train(self):
        for i in range(self._max_steps):
            frac_labeled = self.step()
            print("Successfully finished step %i." % self._step)
            print("Labeled %f percent of unlabeled data." % (100*frac_labeled))
            if frac_labeled < self._min_frac:
                print("This is below the threshold, stopping..")
                return
            if frac_labeled > 0.99:
                print("This is more than 99%, stopping..")
                return
        print("Reached maximum amount of steps.")
        return

    def predict(self):
        classprobs, _, _ = main(self._argv,
                                self._flabel % self._step,
                                self._ftest)
        classes = classprobs[:,0]

        with open("prediction.csv", "w+") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Id","y"])
            for i, predicted_class in enumerate(classes):
                writer.writerow([_OFFSET + i, predicted_class])



def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  # Continuous columns
  x = []
  for i in range(100):
      x += [tf.feature_column.numeric_column('x{}'.format(i+1))]

  # Wide columns and deep columns.
  base_columns = []

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
        n_classes=_NCLASSES)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config,
        n_classes=_NCLASSES)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        n_classes=_NCLASSES)


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


def main(argv, trainfile_name, predictfile_name):
  parser = WideDeepArgParser()
  flags = parser.parse_args(args=argv[1:])

  hidden_units = [int(u) for u in flags.hidden_units.strip("[]").split(",")]

  print("--"*10 + " STATUS " + "--"*10)
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
  train_file = os.path.join(flags.data_dir, trainfile_name)
  predict_file = os.path.join(flags.data_dir, predictfile_name)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return input_fn(train_file, flags.epochs_between_evals, True, flags.batch_size)

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

  prediction = list(model.predict(input_fn=predict_input_fn))

  # Create array w/ [class, probability of this class]
  # TODO Improve: class prob might be low but still much higher than
  # all other probs, that case would also be okay
  predicted_classprobs = np.array([[int(p["classes"][0]), max(p["probabilities"])] for p in prediction])
  return predicted_classprobs, train_file, predict_file


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
        data_dir=_DATA_DIR,
        model_dir=_MODEL_DIR,
        train_epochs=40,
        epochs_between_evals=2,
        batch_size=40)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  trainfile_labeled = "labeled%i.csv"
  trainfile_unlabeled = "unlabeled%i.csv"
  testfile = "test.csv"
  classification_threshold = 0.22

  tc = TClassifier(sys.argv, trainfile_labeled, trainfile_unlabeled, testfile,
                   classification_threshold)

  tc.train()
  tc.predict()
