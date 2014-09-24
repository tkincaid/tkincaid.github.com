import unittest

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.cross_validation import KFold

from ModelingMachine.engine import metrics
from ModelingMachine.engine.gridsearch import BaseGridSearch
from ModelingMachine.engine.gridsearch import ExhaustiveGridSearch
from ModelingMachine.engine.tasks.gbm import _GBC


X, y = make_hastie_10_2(n_samples=1000, random_state=13)


class GridSearchTest(unittest.TestCase):

    GridSearchClass = BaseGridSearch

    def test_single_point(self):
        estimator = DecisionTreeClassifier()
        param_grid = {'max_features': [1.0],
                      'min_samples_leaf': [5],
                      }

        cv = KFold(y.shape[0], random_state=13)
        metric_key = metrics.LOGLOSS
        lower_is_better = metrics.direction_by_name(metric_key)
        score_func = metrics.metric_by_name(metric_key)
        cvgrid = self.GridSearchClass(estimator, param_grid, cv, score_func,
                                      lower_is_better=lower_is_better)
        cvgrid.fit(X, y)
        self.assertEqual(len(cvgrid.grid_scores_), 1)
        print(cvgrid.grid_scores_)
        self.assertEqual(cvgrid.best_params_, {'max_features': 1.0,
                                               'min_samples_leaf': 5})

    def test_multiple_points(self):
        estimator = DecisionTreeClassifier()
        param_grid = {'max_features': [1.0, 0.1],
                      'min_samples_leaf': [5, 100],
                      }

        cv = KFold(y.shape[0], random_state=13)
        metric_key = metrics.LOGLOSS
        lower_is_better = metrics.direction_by_name(metric_key)
        score_func = metrics.metric_by_name(metric_key)
        cvgrid = self.GridSearchClass(estimator, param_grid, cv, score_func,
                                      lower_is_better=lower_is_better)
        cvgrid.fit(X, y)
        self.assertEqual(len(cvgrid.grid_scores_), 4)
        print(cvgrid.grid_scores_)
        self.assertEqual(cvgrid.best_params_, {'max_features': 1.0,
                                               'min_samples_leaf': 5})

    def test_fit_grid(self):
        """Test GridSearch with an estimator that supports _fit_grid. """
        estimator = _GBC(n_estimators=10, max_depth=2)
        param_grid = {'max_depth': [1],
                      'min_samples_leaf': [5],
                      }
        cv = KFold(y.shape[0], random_state=13)
        metric_key = metrics.LOGLOSS
        lower_is_better = metrics.direction_by_name(metric_key)
        score_func = metrics.metric_by_name(metric_key)
        cvgrid = self.GridSearchClass(estimator, param_grid, cv, score_func,
                                      lower_is_better=lower_is_better)
        cvgrid.fit(X, y)
        self.assertEqual(len(cvgrid.grid_scores_), 1)
        print(cvgrid.grid_scores_)
        self.assertEqual(cvgrid.best_params_, {'max_depth': 1.0,
                                               'min_samples_leaf': 5,
                                               'n_estimators': 10})

class ExhaustiveGridSearchTest(unittest.TestCase):

    GridSearchClass = ExhaustiveGridSearch
