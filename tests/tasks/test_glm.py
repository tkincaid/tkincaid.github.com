#########################################################
#
#       Unit Test for GLM task
#
#       Author: Glen Koundry
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import pytest
import numpy as np
from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.glm import GLMC,GLMR,GLMA,GLMB,GLMP,GLMT,GLMG

class TestGLM(BaseTaskTest):

    @pytest.mark.dscomp
    def test_arguments(self):
        self.assertEqual(GLMG.arguments['d']['default'],'0')
        self.assertEqual(GLMP.arguments['d']['default'],'1')
        self.assertEqual(GLMB.arguments['d']['default'],'2')
        self.assertEqual(GLMA.arguments['d']['default'],'3')
        self.assertEqual(GLMT.arguments['d']['default'],'4')

        self.assertEqual(GLMR.arguments['d']['values'],["Gaussian","Poisson","Gamma","Tweedie"])
        self.assertEqual(GLMC.arguments['d']['values'],["Bernoulli"])
        self.assertEqual(GLMR.arguments['tl']['values'],[False,True])

        xt = np.array([[1,2,3],[4,5,6],[7,8,9]])
        yt = np.array([0,1,0])
        self.check_arguments(GLMG,xt=xt,yt=yt)

    def test_GLMR_reproducible(self):
        X,Y,Z = self.create_reg_data()
        replicate_reference = [140.97279499,  189.66740154,  201.70204711]
        t = self.check_task('GLMR',X,Y,Z,transform=True,
                            reference=replicate_reference)

    def test_GLMC_reproducible(self):
        X,Y,Z = self.create_bin_data()
        replicate_reference = [0.91838113,  0.23796596,  0.999]
        t = self.check_task('GLMC',X,Y,Z,transform=True,
                            reference=replicate_reference)

    def test_GLMR_coeffiecients(self):
        X,Y,Z = self.create_bin_data()
        t = self.check_task('GLMC',X,Y,Z)
        expected_coef = [['NumberOfTime60-89DaysPastDueNotWorse', 2.063845], ['NumberOfTimes90DaysLate', 1.955797], ['NumberOfDependents-mi', -1.716182], ['NumberRealEstateLoansOrLines', 0.387903], ['NumberOfDependents', 0.379189], ['NumberOfTime30-59DaysPastDueNotWorse', 0.331658], ['MonthlyIncome-mi', -0.172734], ['NumberOfOpenCreditLinesAndLoans', 0.097224], ['age', -0.033471], ['RevolvingUtilizationOfUnsecuredLines', 0.002597], ['DebtRatio', -0.000175], ['MonthlyIncome', -6.7e-05], ['Unnamed: 0', -6.1e-05]]
        self.assertEqual(expected_coef, t._get_coefficients((0,-1)))

    def test_GLMC_coeffiecients(self):
        X,Y,Z = self.create_reg_data()
        t = self.check_task('GLMR',X,Y,Z)
        expected_coef = [['Var6', -208.207136], ['Var5', 103.846141], ['Var7', 89.563819], ['Var1', -36.894176], ['Var8', 36.730475], ['Var2', 34.169299], ['Var3', 23.402697], ['Var4', 4.997221]]
        self.assertEqual(expected_coef, t._get_coefficients((0,-1)))

    @pytest.mark.dscomp
    def test_basic_gridsearch(self):
        X,Y,Z = self.create_reg_data()
        t = self.check_task('GLMT p=[1.1,1.3,1.5]',X,Y,Z)

    def test_params(self):
        X,Y,Z = self.create_reg_data()
        bm = GLMR('tl=0;d=3;p=[1.1,1.3,1.5]')
        tuneparms,fixedparms = bm._separate_parameters('(0,-1)')
        est = bm._create_estimator(fixedparms)
        self.assertEqual(est.get_params(True),{'tweedie_log': False, 'distribution': 'Tweedie'})
        est.set_params(tweedie_log=True, distribution='Tweedie')
        self.assertEqual(est.get_params(True),{'tweedie_log': True, 'distribution': 'Tweedie'})

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
