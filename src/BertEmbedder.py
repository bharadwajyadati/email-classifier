"""
    Bert/Distil Bert Embedding for nlp Task 
"""
import torch
import numpy as np
from Embedding import BaseEmbedder
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel

"""
        Use this class for embedding intead of one hot encoding or nn.embedding
        in case of NLP tasks for better output 
"""


class BertEmbedding(BaseEmbedder):

    def __init__(self, model='bert'):
        super().__init__()

        if model == 'bert':
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # TODO: need to save to prevent downloading again
        elif model == 'distilbert':
            self.model = DistilBertModel.from_pretrained(
                'distilbert-base-uncased')
            self.tokenziner = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased')

    """
        Tokenizer adds extra tokens like CLS and SEP

        sentence --> CLS <sentence> SEP --> vectorized tokens

        CLS --> classification token 
    """

    def _tokenize(self, df, col):
        return df[col].apply(
            lambda x: self.tokenizer.encode(x, add_special_tokens=True))

    """
        padding them to have equal length 

    """

    def _padding(self, tokens):
        max_len = 0
        for i in tokens.values:
            if len(i) > max_len:
                max_len = len(i)
        return np.array([i + [0]*(max_len-len(i)) for i in tokens.values])

    """
        Masking the padded values with attenions
        Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

    """

    def _masking(self, padded_tokens):
        return np.where(padded_tokens != 0, 1, 0)  # only padded values :)

    """
        TODO: check if we can use device
    """

    def _features(self, padded_tokens, attenion_mask):
        input_ids = torch.tensor(padded_tokens)
        attenion_mask = torch.tensor(attenion_mask)
        with torch.no_grad():
            last_layer_hidden_states = self.model(input_ids, attenion_mask)
        # just taking cls for our classification task
        return last_layer_hidden_states[0][:, 0, :].numpy()

    """
        TODO: make it better
    """

    def feature_extraction(self, df, col):
        padded_tokens = self._padding(self._tokenize(df, col))
        maked_tokens = self._masking(padded_tokens)
        return self._features(padded_tokens, maked_tokens)
