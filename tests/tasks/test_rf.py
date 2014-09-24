#########################################################
#
#       Unit Test for tasks/rf.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest
import numpy as np
import logging

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.rf import RFC, RFI, RFE, RFR

from tesla.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from tesla.ensemble import RandomForestClassifier, RandomForestRegressor

class TestRF(BaseTaskTest):

    def test_positional_args(self):
        task = RFC()
        self.assertEqual( task.parse_positional_args('100;5;0'), 'nt=100;ls=5;e=0')

    @pytest.mark.dscomp
    def test_arguments(self):
        self.assertEqual(RFC.arguments['c']['default'],'0')
        self.assertEqual(RFI.arguments['c']['default'],'0')
        self.assertEqual(RFE.arguments['c']['default'],'1')
        self.assertEqual(RFR.arguments['c']['default'],'0')
        self.assertEqual(RFR.arguments['c']['values'],['mse'])

        for t in  [RFC,RFI,RFE]:
            self.assertEqual(t.arguments['c']['values'],['gini','entropy'])
            self.assertEqual(t.arguments['e']['values'],['RandomForestClassifier','ExtraTreesClassifier'])

        self.assertEqual(RFR.arguments['c']['values'],['mse'])
        self.assertEqual(RFR.arguments['e']['values'],['RandomForestRegressor','ExtraTreesRegressor'])

        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(RFC,xt=xt,yt=yt)

    def test_defaults(self):
        self.check_default_arguments(RFC, ExtraTreesClassifier,['n_estimators','min_samples_leaf','random_state','n_jobs'])
        self.check_default_arguments(RFC, RandomForestClassifier,['n_estimators','min_samples_leaf','random_state','n_jobs','bootstrap'])
        self.check_default_arguments(RFR, ExtraTreesRegressor,['n_estimators','min_samples_leaf','max_features','random_state','n_jobs'])
        self.check_default_arguments(RFR, RandomForestRegressor,['n_estimators','min_samples_leaf','max_features','random_state','n_jobs','bootstrap'])

    def test_RFR_reproducible(self):
        X,Y,Z = self.create_reg_data()
        replicate_reference = [180.79373775,  138.95547328,  223.75394114]
        t = self.check_task('RFR nt=10',X,Y,Z,transform=True,
                            reference=replicate_reference)

    @pytest.mark.dscomp
    def test_basic_gridsearch(self):
        X,Y,Z = self.create_bin_data()
        t = self.check_task('RFC nt=10;e=1;ls=[3,5];c=gini;mf=[1,.5]',X,Y,Z)

    def test_RFC_reproducible(self):
        X,Y,Z = self.create_bin_data()
        replicate_reference = [0.54709123,  0.29244697,  0.7411724]
        t = self.check_task('RFC nt=10',X,Y,Z,transform=True,
                            reference=replicate_reference)
        np.testing.assert_array_almost_equal(t.pred_stack[(0, -1)][:4],
                                             np.array([0.5871266, 0.38076067, 0.70665917, 0.25838417]))

    def test_RFE_reproducible(self):
        X,Y,Z = self.create_bin_data()
        replicate_reference = [0.54709123,  0.29244697,  0.7411724]
        t = self.check_task('RFE nt=10',X,Y,Z,
                            reference=replicate_reference)

    def test_RFI_reproducible(self):
        X,Y,Z = self.create_bin_data()
        replicate_reference = [0.54709123,  0.29244697,  0.7411724]
        t = self.check_task('RFI nt=10',X,Y,Z,
                            reference=replicate_reference)

    @pytest.mark.dscomp
    def test_max_features_gridsearch(self):
        """Test grid search on max_features if mf is too low. """
        X,Y,Z = self.create_bin_data()
        t = self.check_task('RFC nt=1;e=1;c=gini;mf=[0.0001, 0.1, 0.3, 0.8]', X, Y, Z)
        self.assertEquals(t.parameters['max_features'], [1, 0.1, 0.3, 0.8])

    def test_max_features_wo_gridsearch(self):
        """Test max_features if mf is too low w/o gridsearch. """
        X,Y,Z = self.create_bin_data()
        t = self.check_task('RFC nt=1;e=1;c=gini;mf=0.0001', X, Y, Z)
        self.assertEquals(t.parameters['max_features'], 1)


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
