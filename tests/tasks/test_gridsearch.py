#########################################################
#
#       Unit Test for gridsearch and tasks
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2014
#
########################################################
import pytest

from base_task_test import BaseTaskTest

from ModelingMachine.engine import metrics


class TestPatternSearch(BaseTaskTest):
    t_a = 1

    @pytest.mark.dscomp
    def test_GBC(self):
        """Smoke test for classification. """
        X, Y, Z = self.create_bin_data()
        task_args = ['md=[3, 4]', 'ls=[2, 4]', 'lr=0.002', 't_f=0.2',
                     't_m={metrics.AUC}'.format(metrics=metrics)]
        task_args.append('t_a={}'.format(self.t_a))
        task_desc = 'GBC '+';'.join(task_args)
        t = self.check_task(task_desc, X, Y, Z, transform=True,
                            standardize=False)
        # check default of 1
        self.assertEqual(t.gridsearch_parameters['step'], 1)


class TestExhaustiveSearch(TestPatternSearch):
    t_a = 0
