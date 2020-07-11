"""
    Base class for all model training class

"""
import pickle
import numpy as np
import logging
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


class BaseModel(object):

    def __init__(self, stratified=False):

        self.nthread = -1

        self.n_iter = 50
        self.num_folds = 10

        self.classifier = "Dummy"

        self.model_pkl = self.classifier + "model.pkl"

        if stratified:
            self.model = DummyClassifier(strategy="stratified")
        else:
            self.model = DummyClassifier(strategy="most_frequent")

        self.metrics = "auc"  # default metric for classification task

    def hp_grid(self):
        param_grid = {}
        return param_grid

    def fit(self, X_train, y_train, X_test, y_test):

        n_classes = len(np.bincount(y_train))
        if n_classes == 2:
            self.objective = 'binary'
        elif n_classes > 2:
            self.objective = 'multiclass'

        # measure time taken for trainign and save the model
        model = self.model

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = round(accuracy_score(y_test, preds), 5) * 100
        roc = round(roc_auc_score(y_test, preds), 5) * 100

        cls_report = classification_report(y_test, preds)

        print("Results from " + self.classifier)
        print(f"""Accuray: {acc}%
        ROC-AUC: {roc}%""")

        logger.info("results from " + self.classifier)
        logger.info(f"""Accuray: {acc}%
        ROC-AUC: {roc}%""")

        print(cls_report)

        logger.info(cls_report)

        self.save_model(model)
        # save the model
        return acc, roc, cls_report

    def predict(self, rows):
        preds = []
        model = self.restore_model()
        for row in rows:
            preds.append(model.predict(row))  # pylint: disable=no-member
        return preds

    def save_model(self, model):
        pickle.dump(model, open(self.model_pkl, "wb"))

    def restore_model(self):
        self.model = pickle.load(open(self.model_pkl, "rb"))
        return self.model_pkl
