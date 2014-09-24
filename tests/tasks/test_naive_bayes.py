#########################################################
#
#       Unit Test for naive bayes tasks
#
#       Author: Jay and Peter
#
#       Copyright DataRobot, Inc. 2013
#
########################################################
import numpy as np
import pytest

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import make_blobs, load_digits
from sklearn.cross_validation import train_test_split

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.naive_bayes import GaussianNBC, MultinomialNBC, BernoulliNBC
from ModelingMachine.engine.tasks.naive_bayes import CombinerNBClassifier
from ModelingMachine.engine.vertex import Vertex


class TestGNBC(BaseTaskTest):
    """Test case for GaussianNB classifier. """

    def test_arguments(self):
        xt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        yt = np.array([0, 1, 0])
        self.check_arguments(GaussianNBC, GaussianNB, xt, yt)

    def test_classification(self):
        """GaussianNB smoke test."""
        X, Y, Z = self.create_bin_data()
        self.check_task('GNBC', X, Y, Z, transform=False, standardize=False)

    def test_classification_transform(self):
        """Test transform method of GNBC. """
        X, Y, Z = self.create_bin_data()

        # Code copied from check task
        tasks = ['NI']
        vertex = Vertex(tasks, 'id')
        X = vertex.fit_transform(X, Y, Z)
        vertex = Vertex(['GNBC'], 'id')
        vertex.fit(X, Y, Z)
        task, xfunc, yfunc = vertex.steps[-1]

        out = task.transform(X, Y, Z)
        # check if ordering on transform is the same as for joint_log_likelihood
        for p in Z:
            key = (p['r'], p['k'])
            trans_out = out(**p)
            est = task.model[key]
            xt = X(**p)
            probas = est._joint_log_likelihood(xt)

            rank_probas = np.argsort(probas[:, 1])
            rank_trans = np.argsort(trans_out[:, 1])

            np.testing.assert_equal(rank_trans, rank_probas)


class TestBNBC(BaseTaskTest):
    """Test case for BernoulliNB classifier. """

    def test_arguments(self):
        xt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        yt = np.array([0, 1, 0])
        self.check_arguments(BernoulliNBC, BernoulliNB, xt, yt)

    def test_classification(self):
        """GaussianNB smoke test. """
        X, Y, Z = self.create_bin_data()
        self.check_task('BNBC', X, Y, Z, transform=False, standardize=False)


class TestMNBC(BaseTaskTest):
    """Test case for MultinomialNB classifier. """

    def test_arguments(self):
        xt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        yt = np.array([0, 1, 0])
        self.check_arguments(MultinomialNBC, MultinomialNB, xt, yt)

    def test_classification(self):
        """GaussianNB smoke test. """
        X, Y, Z = self.create_bin_data()
        self.check_task('MNBC', X, Y, Z, transform=False, standardize=False)


class TestCNBC(BaseTaskTest):
    """Test case for CombinerNB Classifier. """

    def test_gaussian_combiner(self):
        X, y = make_blobs(1000, 5, 2, random_state=13)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)
        est = GaussianNB()
        est.fit(X_train, y_train)

        comb = CombinerNBClassifier()
        comb.fit(X_train, y_train)

        X_test_prime = est._joint_log_likelihood(X_test)
        X_test_prime -= np.log(est.class_prior_)

        pred_comb = comb.predict_proba(X_test_prime)
        pred = est.predict_proba(X_test)

        np.testing.assert_array_almost_equal(pred, pred_comb)
        error_rate = np.mean(y_test != np.argmax(pred_comb, axis=1))

        # perfect score on test
        self.assertLess(error_rate, 0.00001)

    def test_bernoulli_combiner(self):
        # Non regression test to make sure that any further refactoring / optim
        # of the NB models do not harm the performance on a slightly non-linearly
        # separable dataset
        digits = load_digits()
        X, y = digits.data, digits.target
        binary_3v8 = np.logical_or(digits.target == 3, digits.target == 8)
        X, y = X[binary_3v8], y[binary_3v8]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)

        est = BernoulliNB()
        est.fit(X_train, y_train)

        comb = CombinerNBClassifier()
        comb.fit(X_train, y_train)

        X_test_prime = est._joint_log_likelihood(X_test)
        X_test_prime -= est.class_log_prior_

        pred_comb = comb.predict_proba(X_test_prime)
        pred = est.predict_proba(X_test)

        np.testing.assert_array_almost_equal(pred, pred_comb)

        # at least 0.94 accuracy
        acc = np.mean(y_test == est.classes_[np.argmax(pred_comb, axis=1)])
        self.assertGreater(acc, 0.94)

