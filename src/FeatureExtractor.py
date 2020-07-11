"""
    Should be intelligent Feature Extractor 
"""

import logging
import numpy as np
from Embedding import BaseEmbedder
from BertEmbedder import BertEmbedding
import pandas as pd

from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer

__version__ = 0.1


logger = logging.getLogger(__name__)

# TODO: needs to evaluate perfromance vs complexity

"""
    Generic class for feature extraction for classification tasks

    :param file_path: file_path for reading the csv file ** assumes data in csv format , need to handle with multi input formats 

    :param txt_emb_ids: only for nlp
"""


class GenericFeatureExtractor(object):

    def __init__(self, file_path, txt_emb_ids, out_col, delimiter="~", sample_ratio=1):

        self.file_path = file_path
        self.delimiter = delimiter
        self.out_col = out_col
        self.sample_ratio = sample_ratio
        self.txt_emb_ids = txt_emb_ids
        self.text_column = 'text'  # new field for storing the text
        self.embedding = 'bert'  # should be intelligent enough , change it

    """
        prepare dataframe for tokenziation,basing on the load, try to slice data
    """

    def _prepare_df(self):

        try:
            df = pd.read_csv(self.file_path, delimiter=self.delimiter)
        except Exception as ex:
            logger.error("exception " + ex)
            raise Exception

        temp = ""

        rows = int(df.shape[0] * self.sample_ratio)  # test
        df = df[:rows]

        if self.txt_emb_ids:  # in case to embed long text seqences
            for index in self.txt_emb_ids[::-1]:
                temp = df[df.columns[index]].astype(str) + " " + temp
            df[self.text_column] = temp.str.strip()

        return df

    """
         only used in case of non tree based algoritms
         
    """

    def normalize(self):
        pass

    def analyize_df(self):
        pass
    """
        depending on max sring length , deufalting to bert
    """

    def extract_features(self):
        df = self._prepare_df()
        #columns = df.columns
        num_cols = df._get_numeric_data().columns
        num_cols = num_cols.drop(self.out_col)  # this is output
        y = df[self.out_col]
        # need to see how to include them
        #cat_cols = list(set(columns) - set(num_cols))

        # for given text columns to embedd
        if self.txt_emb_ids:
            embedding = BaseEmbedder()
            if self.embedding == 'bert':
                embedding = BertEmbedding()
            text_embeddings = embedding.feature_extraction(
                df, self.text_column)
            num_embeddings = df[num_cols].to_numpy(
            ).reshape(-1, num_cols.shape[0])
            x = np.hstack((text_embeddings, num_embeddings))
        else:
            pass

        return x, y
