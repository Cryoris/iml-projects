"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import urllib

import numpy


def extract_features(filename, train_dir):
  """Extract the images into a D uint8 numpy array [index, y=1, x, depth=1]."""
  print('Extracting', filename)
  with open(train_dir+filename) as file:
    data = numpy.genfromtxt(file, delimiter=',')
    num_datapoints = len(data)
    rows = 1
    cols = len(data[0])
    data = data.reshape(num_datapoints, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot.astype(int)


def extract_labels(filename, train_dir, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with open(train_dir+filename) as file:
    labels = numpy.genfromtxt(file, delimiter=',').astype(int)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [a, b] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      #print("shape",images.shape)
      for dot in range(len(images[0,:])):
          ma = max(images[:,dot])
          mi = min(images[:,dot])
          
          images[:,dot] -= mi
          images[:,dot] /= (ma-mi) 
       
      #images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled):
        self.n_labeled = n_labeled
        self._images = images

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(images, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        images = images[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(10)[l==1][0] for l in labels])
        idx = indices[y==0][:5] # never used but made problems for eval
        n_classes = y.max() + 1
        n_from_each_class = int(n_labeled / n_classes)
        i_labeled = []
        for c in range(n_classes):
            i = indices[y==c][:n_from_each_class]
            i_labeled += list(i)
        l_images = images[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_images, l_labels)
   
    @property
    def images(self):
        return self._images

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return images, labels

def read_data_sets(train_dir, n_labeled = 100, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets

  TRAIN_IMAGES = "train_labeled_withoutLabel_useTrain.csv"
  TRAIN_LABELS = "train_labeled_justLabel_useTrain.csv"
  TEST_IMAGES = "train_labeled_withoutLabel_useTest.csv"
  TEST_LABELS = "train_labeled_justLabel_useTest.csv"
  EVAL_IMAGES = "test.csv"
  VALIDATION_SIZE = 0

  train_images = extract_features(TRAIN_IMAGES, train_dir)
  train_labels = extract_labels(TRAIN_LABELS, train_dir, one_hot=one_hot)
  test_images = extract_features(TEST_IMAGES, train_dir)
  test_labels = extract_labels(TEST_LABELS, train_dir, one_hot=one_hot)
  
  eval_images = extract_features(EVAL_IMAGES, train_dir)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = SemiDataSet(train_images, train_labels, n_labeled)
  #data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  
  data_sets.eval = DataSet(eval_images, numpy.ones((eval_images.shape[0],10)))

  return data_sets
