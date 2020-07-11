import xgboost
from Model import BaseModel
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class XGB(BaseModel):

    def __init__(self, stratified=False):

        super().__init__(stratified)

        self.random_seed = 42

        self.classifier = "XGBoost"

        # TODO :convert the input vectors in Dmatrix for faster processsing

        self.model = XGBClassifier(n_jobs=-1, objective='binary:logistic',
                                   eval_metric=["error", "auc"])

        # TODO: remove this
        self.objective = 'binary'

    def hp_grid(self):
        # generic parameters for xg boost , needs to be changed or use bayasian
        param_grid = {
            "learning_rate": [0.1, 0.01],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "subsample": [0.6, 0.8, 1.0],
            "max_depth": [2, 3, 4],
            "n_estimators": [100, 200, 300, 400],
            "reg_lambda": [1, 1.5, 2],
            "gamma": [0, 0.1, 0.3],
        }
        #param_grid['njobs'] = self.nthread
        #param_grid['nthread'] = self.nthread

        # if self.objective == 'binary':
        #    param_grid['objective'] = 'binary:logistic'
        #    param_grid['eval_metric'] = ['error', self.metrics]
        # elif self.objective == 'multi':
        #    param_grid['objective'] = 'multi:softmax'
        #    param_grid['eval_metric'] = [self.metrics]

        return param_grid

    def fit(self, X_train, y_train, X_test, y_test):

        scoring = {
            'AUC': 'roc_auc',
            'Accuracy': make_scorer(accuracy_score)
        }

        # measure time taken for trainign and save the model
        # create the Kfold object

        kfold = StratifiedKFold(n_splits=self.num_folds)
        # create the grid search object

        random = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.hp_grid(),
            cv=kfold,
            scoring=scoring,
            # n_jobs=self.nthread,
            n_iter=self.n_iter,
            refit="AUC",
            n_jobs=-1
        )
        # fit grid search
        best_model = random.fit(X_train, y_train)

        self.save_model(best_model)

        print(f'Best score: {best_model.best_score_}')
        print(f'Best model: {best_model.best_params_}')

        pred_test = best_model.predict(X_test)
        pred_train = best_model.predict(X_train)
        print('Train Accuracy: ', accuracy_score(y_train, pred_train))
        print('Test Accuraccy: ', accuracy_score(y_test, pred_test))
        print('\nConfusion Matrix:')
        print(confusion_matrix(y_test, pred_test))
        print('\nClassification Report:')
        print(classification_report(y_test, pred_test))
