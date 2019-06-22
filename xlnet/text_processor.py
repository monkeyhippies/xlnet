"""
text processor to
1. create input example
2. preprocess and tokenize
3. create feature
"""
import tensorflow as tf
import sentencepiece as spm
from tqdm import tqdm

from xlnet.classifier_utils import convert_single_example
from xlnet.prepro_utils import preprocess_text, encode_ids

class TextProcessor(object):

    def __init__(self, spiece_model_file, label_list,
        max_seq_length=128, uncased=False
    ):

        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.uncased = uncased

        self._spiece_model_file = spiece_model_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self._spiece_model_file)

    def _tokenize_wrapper(self):
        uncased = self.uncased
        sp = self.sp
        def tokenize_fn(text):
            text = preprocess_text(text, lower=uncased)
            return encode_ids(sp, text)
        
        return tokenize_fn

    def tokenize(self, example):
        """
        example is an instance of InputExample
        """
        return example

    def process(self, input_examples):

        features = list()
        tokenize_fn = self._tokenize_wrapper()
        for (ex_index, example) in tqdm(enumerate(input_examples)):

            feature = convert_single_example(
                ex_index,
                example,
                self.label_list,
                self.max_seq_length,
                tokenize_fn
            )

            features.append(feature)

        return features

