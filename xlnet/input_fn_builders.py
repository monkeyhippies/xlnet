import tensorflow as tf
from xlnet.tf_record_utils import parser_builder

def tf_record_input_fn_builder(filenames, seq_length, batch_size, shuffle_size=1000):
    def input_fn():
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "input_ids": [feature.input_ids for feature in features],
                    "input_mask": [feature.input_mask for feature in features],
                    "segment_ids": [feature.segment_ids for feature in features],
                    "label_id": [feature.label_id for feature in features]
                },
                train["target"]
            )
        )
        """
        # Import MNIST data
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.shuffle(shuffle_size).repeat().batch(batch_size)
        dataset = dataset.map(parser_builder(seq_length))

        return dataset
    return input_fn
