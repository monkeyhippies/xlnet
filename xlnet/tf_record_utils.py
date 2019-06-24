from tqdm import tqdm
import tensorflow as tf

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_list_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def parser_builder(seq_length):
    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_example(
            serialized_example,
            features={
                'input_ids': tf.FixedLenFeature([seq_length], tf.int64),
                'input_mask': tf.FixedLenFeature([seq_length], tf.int64),
                'segment_ids': tf.FixedLenFeature([seq_length], tf.int64),
                'label_id': tf.VarLenFeature(tf.int64),
                'target': tf.FixedLenFeature([], tf.float32)
            })
        target = features.pop("target")
        return features, target
    return parser

def serialize_example(feature, target):
    """
    Creates a tf.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.

    feature = {
        'input_ids': _int64_list_feature(feature.input_ids),
        'input_mask': _int64_list_feature(feature.input_mask),
        'segment_ids': _int64_list_feature(feature.segment_ids),
        'label_id': _int64_list_feature(feature.label_id),
        'target': _float_feature(target)
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord_file(filename, features, targets):
    # Write the `tf.Example` observations to the file.
    with tf.python_io.TFRecordWriter(filename) as writer:
        for feature, target in tqdm(zip(features, targets)):
            example = serialize_example(feature, target)
            writer.write(example)
