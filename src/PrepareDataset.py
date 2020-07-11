import logging
import pandas as pd

"""
    Generic method to read email content and give it to classifier model

"""


logger = logging.getLogger(__name__)


class Dataloader(object):

    """
        file_path can be local or s3
    """

    def __init__(self, file_path, delimiter=","):
        self.file_path = file_path
        self.delimiter = delimiter
        try:
            self.df = pd.read_csv(
                self.file_path, delimiter=self.delimiter)
        except Exception as e:
            logger.error("failed while parsing file " + e)

    def preprocess(self):
        pass


class EmailLoader(Dataloader):

    def __init__(self, file_path, delimiter=","):
        super().__init__(file_path, delimiter)
        # typical , needs to be changed in case of anomolies
        self.preprocess_columns = ['from', 'subject', 'body']

    def preprocess(self):
        columns = []
        for column in self.df.columns:
            if column.lower() in self.preprocess_columns:
                columns.append(column)
        return columns


