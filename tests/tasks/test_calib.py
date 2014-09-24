############################################
#
# Unit tests for prediction calibration
#
# Author: Xavier Conort
#
# Copyright: 2014
#
############################################
import numpy as np
import pandas as pd
import unittest
import cPickle as pickle

from scipy import sparse
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.naive_bayes import MultinomialNB

from base_task_test import BaseTaskTest
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.response import Response
from ModelingMachine.engine.tasks.calib import CALIB
from ModelingMachine.engine.tasks.calib import sigmoid_calibration, _SigmoidCalibration
from ModelingMachine.engine.tasks.calib import PlattCalibrationTask
from ModelingMachine.engine.tasks.calib import IsotonicRegressionCalibrationTask
from ModelingMachine.engine.tasks.calib import IsotonicCalibrationCVEstimator


class TestCalib(BaseTaskTest):

    def _create_test_data(self):
        X, y = datasets.make_friedman1(n_samples=20, random_state=13)
        X = pd.DataFrame(X)
        Y = Response.from_array(y / y.max())
        Z = Partition(size=X.shape[0], folds=5, reps=1, total_size=X.shape[0])
        Z.set(max_reps=1, max_folds=0)
        return Container(X), Y, Z

    def test_link(self):
        X, Y, Z = self._create_test_data()
        for est in CALIB.arguments['e']['values']:
            for family in CALIB.arguments['f']['values']:
                task_desc = 'CALIB f={family};e={est};pp_stk=0'.format(family=family, est=est)
                t = self.check_task(task_desc, X, Y, Z, transform=False,
                                    standardize=False)
                self.assertEqual(t.parameters['estimator'], est)
                if t.parameters['estimator'] == 'GLM' and family == 'binomial':
                    self.assertEqual(t.parameters['family'], 'Bernoulli')
                elif t.parameters['estimator'] == 'GLM':
                    self.assertEqual(t.parameters['family'], family.title())
                else:
                    self.assertEqual(t.parameters['family'], family)
                desired_link = {'Gamma': 'log',
                                'gaussian': 'identity',
                                'binomial': 'logit',
                                }.get(family, 'log')
                self.assertEqual(t.parameters['link'], desired_link)

                desired_poly_dg = {'GAM': 1, 'GLM': 2}[est]
                self.assertEqual(t.parameters['poly_dg'], desired_poly_dg)

    def test_freq_sev_task_fit(self):
        """
        Test task fit function
        """
        x = np.random.randint(1, 8, (3000, 1))
        xcont = Container()
        xcont.add(x)
        Z = Partition(3000, folds=5, reps=5, total_size=3000)
        Z.set(max_folds=0, max_reps=1)
        ydata = Response.from_array(x[:, 0])
        est = CALIB()
        out = est.fit(xcont, ydata, Z)
        self.assertIsInstance(out, CALIB)
        est = CALIB('f=Gamma;e=GAM')
        out = est.fit(xcont, ydata, Z)
        self.assertIsInstance(out, CALIB)
        est = CALIB('f=poisson;e=GLM;p=2')
        out = est.fit(xcont, ydata, Z)
        self.assertIsInstance(out, CALIB)
        est = CALIB('f=poisson;e=GLM;p=3')
        out = est.fit(xcont, ydata, Z)
        self.assertIsInstance(out, CALIB)

    def test_freq_sev_task_predict(self):
        """
        Test task predict function
        """
        x = np.random.randint(1, 8, (3000, 1))
        ydata = Response.from_array(x[:, 0])
        xcont = Container()
        xcont.add(x)
        Z = Partition(3000, folds=5, reps=5, total_size=3000)
        Z.set(max_folds=0, max_reps=1)
        est = CALIB()
        est = est.fit(xcont, ydata, Z)
        x2 = np.random.random((300, 1))
        x2cont = Container()
        x2cont.add(x2)
        Z = Partition(300, folds=5, reps=5, total_size=300)
        Z.set(max_folds=0, max_reps=1)
        ydata = Response.from_array(x2[:, 0])
        out = est.predict(x2cont, ydata, Z)
        self.assertIsInstance(out, Container)
        for p in out:
            self.assertEqual(out(**p).shape[1], 1)
            self.assertEqual(out(**p).shape[0], x2.shape[0])
            desired = x2
            np.testing.assert_allclose(desired.ravel(), out(**p).ravel())


def brier_score_loss(y_true, y_prob, sample_weight=None):
    """Compute the Brier score
    The smaller the Brier score, the better, hence the naming with "loss".
    Across all items in a set N predictions, the Brier score measures the
    mean squared difference between (1) the predicted probability assigned
    to the possible outcomes for item i, and (2) the actual outcome.
    Therefore, the lower the Brier score is for a set of predictions, the
    better the predictions are calibrated. Note that the Brier score always
    takes on a value between zero and one, since this is the largest
    possible difference between a predicted probability (which must be
    between zero and one) and the actual outcome (which can take on values
    of only 0 and 1).
    The Brier score is appropriate for binary and categorical outcomes that
    can be structured as true or false, but is inappropriate for ordinal
    variables which can take on three or more values (this is because the
    Brier score assumes that all possible outcomes are equivalently
    "distant" from one another).
    Parameters
    ----------
    y_true : array, shape (n_samples,)
    True targets.
    y_prob : array, shape (n_samples,)
    Probabilities of the positive class.
    sample_weight : array-like of shape = [n_samples], optional
    Sample weights.
    Returns
    -------
    score : float
    Brier score
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import brier_score_loss
    >>> y_true = [0, 1, 1, 0]
    >>> y_prob = [0.1, 0.9, 0.8, 0.3]
    >>> brier_score_loss(y_true, y_prob) # doctest: +ELLIPSIS
    0.037...
    >>> brier_score_loss(y_true, np.array(y_prob) > 0.5)
    0.0
    References
    ----------
    http://en.wikipedia.org/wiki/Brier_score
    """
    return np.average((y_true - y_prob) ** 2, weights=sample_weight)


def calibration_curve(y_true, y_prob, normalize=False, n_bins=5):
    """Compute true and predicted probabilities for a calibration curve

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_prob : array, shape (n_samples,)
        Probabilities of the positive class.

    normalize : bool, optional, default=False
        Whether y_prob needs to be normalized into the bin [0, 1], i.e. is not
        a proper probability. If True, the smallest value in y_prob is mapped
        onto 0 and the largest one onto 1.

    n_bins : int
        Number of bins. A bigger number requires more data.

    Returns
    -------
    prob_true : array, shape (n_bins,)
        The true probability in each bin (fraction of positives).

    prob_pred : array, shape (n_bins,)
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    """

    if normalize:  # Normalize predicted values into interval [0, 1]
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    elif y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1] and normalize is "
                         "set to False.")

    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

    binids = np.digitize(y_prob, bins) - 1
    ids = np.arange(len(y_true))

    u_binids = np.unique(binids)  # don't consider empty bins

    prob_true = np.empty(len(u_binids))
    prob_pred = np.empty(len(u_binids))

    for k, binid in enumerate(u_binids):
        sel = ids[binids == binid]
        prob_true[k] = np.mean(y_true[sel])
        prob_pred[k] = np.mean(y_prob[sel])

    return prob_true, prob_pred


class TestCalibratedClassifier(BaseTaskTest):
    # Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
    # License: BSD 3 clause

    def test_calibration(self):
        """Test calibration objects with isotonic and sigmoid"""
        n_samples = 5 * 500
        X, y = make_classification(n_samples=2 * n_samples, n_features=6,
                                   random_state=42)

        X -= X.min()  # MultinomialNB only allows positive X

        # split train and test
        X_train, y_train = X[:n_samples], y[:n_samples]
        X_test, y_test = X[n_samples:], y[n_samples:]

        # Naive-Bayes
        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        X_test = clf.predict_proba(X_test)[:, 1]

        # Naive Bayes with calibration
        for this_X_train, this_X_test in [(X_train, X_test),
                                          (sparse.csr_matrix(X_train),
                                          sparse.csr_matrix(X_test.reshape((-1, 1))))]:
            for calib in [PlattCalibrationTask(), IsotonicRegressionCalibrationTask()]:
                Xcont = Container()
                print this_X_test.shape
                Xcont.add(this_X_test)
                Z = Partition(X_test.shape[0], folds=1, reps=0)

                Z.set(max_reps=0, max_folds=0)
                calib.fit(Xcont, y_test, Z)

                prob_pos_pc_clf = calib.predict(Xcont, y_test, Z)

                for p in Z:
                    print p
                calib_probs = prob_pos_pc_clf(r=-1, k=-1).flatten()
                print calib_probs[:10]
                print X_test[:10]
                print np.corrcoef(np.vstack((calib_probs, X_test)))
                print calib_probs.min()
                print calib_probs.max()
                print "Calibrated histogram: "
                print np.histogram(calib_probs, 10)
                print "Uncalibrated histogram: "
                print np.histogram(X_test, 10)
                self.assertGreater(brier_score_loss(y_test, X_test),
                                   brier_score_loss(y_test, calib_probs))

    def test_sigmoid_calibration(self):
        """Test calibration values with Platt sigmoid model"""
        exF = np.array([5, -4, 1.0])
        exY = np.array([1, -1, -1])
        # computed from my python port of the C++ code in LibSVM
        AB_lin_libsvm = np.array([-0.20261354391187855, 0.65236314980010512])
        np.testing.assert_array_almost_equal(AB_lin_libsvm, sigmoid_calibration(exF, exY), 3)
        lin_prob = 1. / (1. + np.exp(AB_lin_libsvm[0] * exF + AB_lin_libsvm[1]))
        sk_prob = _SigmoidCalibration().fit(exF, exY).predict(exF)
        np.testing.assert_array_almost_equal(lin_prob, sk_prob, 6)

    def test_calibration_curve(self):
        """Check calibration_curve function"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0., 0.1, 0.2, 0.8, 0.9, 1.])
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=2)
        prob_true_unnormalized, prob_pred_unnormalized = \
            calibration_curve(y_true, y_pred * 2, n_bins=2, normalize=True)
        self.assertEqual(len(prob_true), len(prob_pred))
        self.assertEqual(len(prob_true), 2)
        np.testing.assert_array_almost_equal(prob_true, [0, 1])
        np.testing.assert_array_almost_equal(prob_pred, [0.1, 0.9])
        np.testing.assert_array_almost_equal(prob_true, prob_true_unnormalized)
        np.testing.assert_array_almost_equal(prob_pred, prob_pred_unnormalized)

    def test_isotonic_pickle(self):
        x = np.random.normal(loc=0.0, size=100).reshape((100, 1))
        y = (x > 0.0).ravel()
        cal = IsotonicCalibrationCVEstimator(cv=3)
        cal.fit(x, y)
        cal_ser = pickle.dumps(cal, pickle.HIGHEST_PROTOCOL)
        cal2 = pickle.loads(cal_ser)
        np.testing.assert_array_equal(cal2.predict_proba(x), cal.predict_proba(x))


if __name__ == '__main__':
    unittest.main()
