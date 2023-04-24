import tensorflow as tf
import numpy as np


class DataSetLoader:
    def __init__(self, path) -> None:
        with tf.device("/cpu:0"):
            raw_dataset = load_verification_dataset(path)
            list_dataset = list(raw_dataset)
            self.dataset = np.asarray(list(map(lambda x: (x[0].numpy(), x[1].numpy()), list_dataset)))
            self.labels = np.asarray(list(map(lambda x: (x[2]), list_dataset)))

    def __call__(self) -> list:
        return self.dataset, self.labels

    def get_pair(self, pair_id) -> list:
        return self.dataset[pair_id]


def load_verification_dataset(path: str) -> list:
    """load benchmark dataset from tfrecord

    :return: tuple containing images and labels (complete dataset)
    """

    def _parse_samples(tfrecord) -> list:
        """Parsing a tfrecord example (pairs of images and corresponding label)

        :param tfrecord: the raw tf record file
        :return: decoded images and labels
        """
        features = {
            "image1": tf.io.FixedLenFeature([], tf.string),
            "image2": tf.io.FixedLenFeature([], tf.string),
            "issame": tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(tfrecord, features)
        x1 = tf.image.decode_png(sample["image1"])
        x1 = tf.image.convert_image_dtype(x1, dtype=tf.float32)
        x2 = tf.image.decode_png(sample["image2"])
        x2 = tf.image.convert_image_dtype(x2, dtype=tf.float32)
        y = tf.cast(sample["issame"], tf.bool)
        return x1, x2, y

    raw_dataset = tf.data.TFRecordDataset(path)
    dataset = raw_dataset.map(_parse_samples)
    return dataset.take(-1)
