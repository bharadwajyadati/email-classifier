import lightgbm
from Model import BaseModel


class LGM(BaseModel):

    def __init__(self,  stratified=False):
        super().__init__(stratified)

        self.classifier = "LightGBM"
        self.categorical_feature = 'auto'
        self.stratified = stratified

    def hp_grid(self):
        param_grid = {
            'boosting': 'gbdt',
            'num_leaves': 31,  # 2^n -1
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.01,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            # 'verbose': 0,
            # "eval_set": [(self.X_test, self.y_test)],
            # 'eval_names': ['valid'],
        }
        if self.objective == 'binary':
            param_grid['objective'] = 'binary'
            param_grid['application'] = 'binary'
        elif self.objective == 'multi':
            param_grid['objective'] = 'multiclass'
        param_grid['eval_metric'] = self.metrics
        if self.stratified:
            param_grid['is_unbalance'] = 'true'
        return param_grid

    def fit(self, X_train, y_train, X_test, y_test):

        train_data = lightgbm.Dataset(X_train, label=y_train)
        test_data = lightgbm.Dataset(X_test, label=y_test)
        evals_results = {}
        model = lightgbm.train(self.hp_grid(),
                               train_data,
                               valid_sets=test_data,
                               num_boost_round=5000,
                               early_stopping_rounds=100,
                               evals_result=evals_results,
                               categorical_feature=self.categorical_feature
                               )
        n_estimators = model.best_iteration
        evals_results['valid'][self.metrics][n_estimators-1]
        print("n_estimators : ", n_estimators)
        print(self.metrics+":", evals_results['valid']
              [self.metrics][n_estimators-1])
