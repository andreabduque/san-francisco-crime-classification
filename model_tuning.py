from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, make_scorer
import pandas as pd

class ClassifierPipeline:
    def __init__(self, classifier, classifier_parameters, n_jobs=-1):
        self.pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', classifier),
        ])

        self.parameters = {}
        for key, value in classifier_parameters.items():
            self.parameters.update({"clf__" + key: value})

        #By default, 3-fold cv is used
        self.grid_search = None
        self.scorer = make_scorer(log_loss, greater_is_better = False, needs_proba=True)
        self.grid_search = GridSearchCV(self.pipeline, self.parameters, n_jobs=n_jobs, scoring=self.scorer, return_train_score=False)


    def fit(self, data, target):
        self.grid_search.fit(data, target)
        print("Best score: %0.3f" % self.grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = self.grid_search.best_estimator_.get_params()
        for param_name in sorted(self.parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

    def get_cv_results(self):
        return  pd.DataFrame(self.grid_search.cv_results_)