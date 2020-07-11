import os.path
from Model import BaseModel
from LightModel import LGM
from XGModel import XGB
from FeatureExtractor import GenericFeatureExtractor
from sklearn.model_selection import train_test_split


class Pipeline(object):

    def __init__(self, file_path, txt_emb_ids, out_col, cat_cols="", model="lightgbm", delimiter="~", sample_ratio=1):
        self.file_path = file_path
        self.txt_emb_ids = txt_emb_ids
        self.out_col = out_col
        self.delimiter = delimiter
        self.sample_ratio = sample_ratio

        self.cat_cols = cat_cols

        self.ouput_file = "out.txt"

        self.random_state = 42  # default

        self.model = BaseModel(stratified=True)

        if model == "lightgbm":  # default
            self.model = LGM(stratified=True)
        elif model == 'xgboost':
            self.model = XGB(stratified=True)

    def train(self):
        fe = GenericFeatureExtractor(
            self.file_path, self.txt_emb_ids, self.out_col, self.delimiter, self.sample_ratio)
        x, y = fe.extract_features()
        # by default stratify ?  , verify once
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, stratify=y, random_state=self.random_state)

        # by default calling light gbm , needs to be tweaked

        self.model.fit(X_train, y_train, X_test,
                       y_test)  # training

        try:
            f = open(self.ouput_file, "a")
            f.write("training pointer")
        except IOError:
            print("modelling is done , saving pointer")
        finally:
            f.close()

    def predict(self, test):
        try:
            f = open(self.ouput_file, "r")
            self.model.predict(test)
        except IOError:
            print("modelling is not done/not found, try retraining")
        finally:
            f.close()


# if __name__ == "main":
