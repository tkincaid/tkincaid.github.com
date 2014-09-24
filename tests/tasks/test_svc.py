#########################################################
#
#       Unit Test for tasks/svc.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import logging
import numpy as np

from mock import patch
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.svc import LSVC, SVMC, SVMR, SVCR, SVCL, SVCP, SVCS
from ModelingMachine.engine.tasks.svc import ASVMC, ASVMR, ApproxSVC, ApproxSVR
from common.exceptions import ConvergenceError

class TestSVC(BaseTaskTest):

    @pytest.mark.dscomp
    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(LSVC, LinearSVC, xt,yt)
        self.check_arguments(SVMC, SVC, xt,yt)
        self.check_arguments(SVCR, SVC, xt,yt)
        self.check_arguments(SVCL, SVC, xt,yt)
        self.check_arguments(SVCP, SVC, xt,yt)
        self.check_arguments(SVCS, SVC, xt,yt)
        self.check_arguments(SVMR, SVR, xt,yt)

    def test_defaults(self):
        self.check_default_arguments(SVMC, SVC,['C','gamma','random_state','probability'])
        self.check_default_arguments(SVMR, SVR,['C','gamma','random_state'])

    @pytest.mark.dscomp
    def test_01(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('SVMC',X,Y,Z,transform=False,standardize=True)

    @pytest.mark.dscomp
    def test_02(self):
        X,Y,Z = self.create_reg_data()
        self.check_task('SVMR',X,Y,Z,transform=False,standardize=True)

    @pytest.mark.dscomp
    def test_03(self):
        #SVM has a transform method when kernel = linear (k=0)
        X,Y,Z = self.create_bin_data()
        self.check_task('SVMC k=0;mi=99',X,Y,Z,transform=True,standardize=True)

    @pytest.mark.dscomp
    def test_04(self):
        X,Y,Z = self.create_bin_data()
        self.check_task('LSVC',X,Y,Z,transform=True,standardize=True,predtype=np.float64)

    def test_liblin_defaults(self):
        """test defaults of liblinear SVM. """
        # we need to ignore intercept_scaling because sklearn has int default (1) although
        # the scaling should be float only (we use 1.0).
        self.check_default_arguments(LSVC, LinearSVC,['C', 'random_state', 'intercept_scaling'])


class ASVMTest(BaseTaskTest):

    def test_arguments(self):
        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])

        self.check_arguments(ASVMC, ApproxSVC, xt,yt)
        self.check_arguments(ASVMR, ApproxSVR, xt,yt)

    @pytest.mark.dscomp
    def test_approx_kernel_task_clf(self):
        X,Y,Z = self.create_bin_data()
        t = self.check_task('ASVMC', X, Y, Z, transform=False, standardize=True)
        self.assertTrue(isinstance(t.model.values()[0].est.steps[-1][1], LogisticRegression))

    @pytest.mark.dscomp
    def test_approx_kernel_clf(self):
        """Make sure approximate kernel gives same results if n_components covers whole DS. """
        X, y = make_classification(n_samples=300, n_features=10,
                                   weights=[0.833, 0.167], random_state=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)

        clf1 = SVC(kernel='rbf', gamma=0.1, C=1.0, random_state=13).fit(X_train, y_train)
        for approx in ['nystroem', 'fourier']:
            clf2 = ApproxSVC(approx=approx, n_components=X_train.shape[0],
                             loss='l2',
                             gamma=0.1, C=1.0, random_state=13).fit(X_train, y_train)
            np.testing.assert_almost_equal(clf1.score(X_test, y_test),
                                           clf2.score(X_test, y_test), decimal=1)

    @patch('ModelingMachine.engine.tasks.svc.kernel_approximation.Nystroem.fit')
    def test_asvm_fourier_fallback(self, mock_nystroem_fit):
        """Test if we fall back to Fourier Approx upon LinAlgError in Nystroem.fit. """
        mock_nystroem_fit.side_effect = np.linalg.LinAlgError()
        clf = ApproxSVC(approx='nystroem', n_components=10, random_state=13)
        X, y = make_classification(n_samples=300, n_features=10,
                                   weights=[0.833, 0.167], random_state=2)

        clf.fit(X, y)
        # fall back to
        self.assertEqual(clf.approx, 'fourier')


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
