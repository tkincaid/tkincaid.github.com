############################################################################
#
#       unit test for rrf_bm.py (R randomForest)
#
#       Author: Glen Koundry / Mark Steadman / Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################
import unittest
import pytest
import cPickle
import tempfile
from mock import Mock, patch, DEFAULT
import numpy as np

from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.tasks.rrf_bm import RRFEstimatorR, RRFEstimatorC, RRFBMC, RRFBMR
from sklearn.datasets import make_classification, make_regression
import ModelingMachine


class TestRRFEstimatorR(unittest.TestCase):
    """Test R randomForest against sklearn RandomForest """

    @pytest.mark.dscomp
    def test_rrf_vs_sklearn_reg(self):
        """Test R vs. sklearn on boston housing dataset. """
        from sklearn.datasets import load_boston
        from sklearn.cross_validation import train_test_split
        from sklearn.metrics import mean_squared_error
        from sklearn.ensemble import RandomForestRegressor

        boston = load_boston()
        X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                            test_size=0.2, random_state=13)

        n_samples, n_features = X_train.shape
        mtry = int(np.floor(0.3 * n_features))
        # do 100 trees
        r_rf = RRFEstimatorR(**{'ntree': 100, 'nodesize': 1, 'replace': 0,
                                'mtry': mtry, 'corr.bias': False,
                                'sampsize': n_samples, 'random_state': 1234})
        r_rf.fit(X_train, y_train)
        y_pred = r_rf.predict(X_test)
        r_mse = mean_squared_error(y_test, y_pred)

        p_rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, bootstrap=False,
                                     max_features=mtry, random_state=1)
        p_rf.fit(X_train, y_train)
        y_pred = p_rf.predict(X_test)
        p_mse = mean_squared_error(y_test, y_pred)
        print('%.4f vs %.4f' % (r_mse, p_mse))
        # should be roughly the same (7.6 vs. 7.2)
        np.testing.assert_almost_equal(r_mse, p_mse, decimal=0)

    def test_rrf_vs_sklearn_clf(self):
        """Test R vs. sklearn on iris dataset. """
        from sklearn.datasets import load_iris
        from sklearn.cross_validation import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import RandomForestClassifier

        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                            test_size=0.2, random_state=13)

        n_samples, n_features = X_train.shape
        mtry = int(np.floor(0.3 * n_features))
        # do 100 trees
        r_rf = RRFEstimatorC(**{'ntree': 100, 'nodesize': 1, 'replace': 0,
                                'mtry': mtry, 'corr.bias': False,
                                'sampsize': n_samples, 'random_state': 1234})
        r_rf.fit(X_train, y_train)
        y_pred = r_rf.predict(X_test)
        r_acc = accuracy_score(y_test, y_pred)

        p_rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, bootstrap=False,
                                      max_features=mtry, random_state=12)
        p_rf.fit(X_train, y_train)
        y_pred = p_rf.predict(X_test)
        p_acc = accuracy_score(y_test, y_pred)
        print('%.4f vs %.4f' % (r_acc, p_acc))
        np.testing.assert_almost_equal(r_acc, p_acc, decimal=4)


class TestRRF_Regressions(unittest.TestCase):

    def test_can_pickle(self):
        """Test that can pickle. Important since some rpy objects cannot be pickled and good to test explicitly
        so that if they can they can be detected without running the entire test suite."""
        from sklearn.datasets import load_boston
        from sklearn.cross_validation import train_test_split
        boston = load_boston()
        X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target,
                                                            test_size=0.1, random_state=13)

        n_samples, n_features = X_train.shape
        # do 100 trees and turn off fx-subsampling and training set subsampling
        r_rf = RRFEstimatorR(**{'ntree': 10, 'nodesize': 1, 'replace': False,
                                'mtry': n_features, 'corr.bias': False,
                                'do.trace': True, 'sampsize': n_samples, 'random_state': 1234})
        r_rf.fit(X_train, y_train)
        with tempfile.SpooledTemporaryFile(mode='w+b') as tf:
            cPickle.dump(r_rf, tf)

    def test_train_dne_response(self):
        """Test what happens when the training set size dne the response size"""
        X, Y = make_classification(n_classes=2, n_features=25, n_samples=50)
        Y = Y[1:(Y.shape[0]-10)]
        Cont = Container()
        Cont.add(X)
        Z = Partition(X.shape[0],total_size=X.shape[0])
        r_rf = RRFBMC("mf=0.7;nt=10")
        # Cont and Y mismatch
        with self.assertRaisesRegexp(ValueError, 'shape mismatch.+') :
            r_rf.fit(Cont, Y, Z)

    def test_empty_training_set(self):
        """Test what happens when X is an empty container"""
        ContX = Container()
        ContX.add(np.array([]))
        y = np.array([1, 0, 1, 0, 1])
        Z = Partition(5,total_size=5)
        r_rf = RRFBMC("mf=n/3;im=perm;ss=auto;bs=1;nt=10")
        with self.assertRaises(ValueError):
            r_rf.fit(ContX, y, Z)  # in restimator, get_dataFrame

    @pytest.mark.dscomp
    def test_all_classlabels_the_same(self):
        """Test what happens when all the response variables are of the same class"""
        X, Y = make_classification(n_classes=2, n_features=25, n_samples=50)
        Y = np.ones(shape=(X.shape[0], 1))
        Cont = Container()
        Cont.add(X)
        Z = Partition(X.shape[0],total_size=X.shape[0])
        r_rf = RRFBMC(args="mf=sqrt;im=gini;ss=auto;bs=0;nt=10")
        with self.assertRaises(RuntimeError):
            r_rf.fit(Cont, Y, Z)  # In restimator

    def test_all_regression_response_same(self):
        """Test what happens when all the response variables are the same in a regression problem"""
        X, Y = make_regression(n_informative=10, n_features=25, n_samples=50, n_targets=1)
        Y = np.ones(shape=(X.shape[0], 1))
        Cont = Container()
        Cont.add(X)
        Z = Partition(X.shape[0],total_size=X.shape[0])
        r_rf = RRFBMC(args="mf=sqrt;im=gini;ss=auto;bs=0;nt=10")
        with self.assertRaises(RuntimeError):
            r_rf.fit(Cont, Y, Z)  # In restimator

    @pytest.mark.dscomp
    @patch.object(ModelingMachine.engine.tasks.base_modeler.BaseModeler,'_check_stack')
    def test_predict_empty(self,*args):
        """Test what happens when it passes new data as an empty container"""
        X, Y = make_regression(n_informative=10, n_features=25, n_samples=250, n_targets=1)
        Cont = Container()
        Cont.add(X)
        Z = Partition(X.shape[0],total_size=X.shape[0])
        r_rf = RRFBMC(args="mf=sqrt;im=gini;ss=auto;bs=0;nt=10")
        r_rf.fit(Cont, Y, Z)  # In restimator
        ContPredict = Container()
        ZPredict = Partition(3,total_size=3)
        yPred = np.array([])
        yPred2 = np.array([10, 20, 30])
        ZPredict2 = Partition(3,total_size=3)
        with self.assertRaises(ValueError):  # In rpy2 numpy2ri
            r_rf.predict(ContPredict, yPred, ZPredict)
        with self.assertRaises(ValueError):  # In restimator get_dataFrame
            r_rf.predict(ContPredict, yPred2, ZPredict2)

    @pytest.mark.dscomp
    def test_fit_single_col(self):
        """Test that it handles a single column correctly
        """
        x = np.random.randn(500).reshape((500, 1))
        y = np.random.randint(20, size=500)
        X = Container()
        X.add(x)
        r_rf = RRFBMR('nt=10')
        Z = Partition(x.shape[0],total_size=x.shape[0])
        res = r_rf.fit(X, y, Z)

    def test_rrfc_auto(self):
        """Regression test for mtry = auto (should produce
        ncols / 3
        """
        x = np.random.randn(84).reshape((-1, 12))
        y = np.random.randint(0, 2, size=x.shape[1])
        X = Container()
        X.add(x)
        r_rf = RRFBMC()
        Z = Partition(x.shape[0], folds=5, reps=1,total_size=x.shape[0])
        Z.set(max_reps=1, max_folds=0)
        r_rf._modify_parameters(X(), y)
        self.assertEqual(r_rf.parameters['mtry'],  4)


if __name__ == '__main__':
    unittest.main()
