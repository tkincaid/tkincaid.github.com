############################################################################
#
#       unit test for Tasks
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################
import os
import pandas
import numpy as np
import sys
import unittest
import tempfile
import cPickle
import hashlib
import logging
import traceback
from copy import deepcopy

tests_dir = os.path.dirname(os.path.abspath(__file__) )
modeling_machine_dir = os.path.join(tests_dir, '../..') 
sys.path.append(modeling_machine_dir)

from ModelingMachine.engine.vertex import task_map
from ModelingMachine.engine.controller import Controller
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.pandas_data_utils import getX, getY, varTypes
from ModelingMachine.engine.tasks.standardize import Standardize

import ModelingMachine.engine.tasks.converters


def get_default_data():
    # Create a binary classification task with some useless dimensions
    imputer = ModelingMachine.engine.tasks.converters.Numeric_impute()

    eps = 1e-8

    n_samples = 200
    n_good_dims = 5
    n_noise_dims = 5
    randseed = 1
    np.random.seed(randseed)

    x = np.random.randn(n_samples, n_good_dims)
    xbad = np.random.randn(n_samples,n_noise_dims)
    xfull = np.hstack((x, xbad))
    xfull = (xfull - xfull.min() + eps ) / (xfull.max() - xfull.min() )
    xfull = xfull * 0.998


    trans_mat = np.random.randn(n_good_dims, 1)
    y = np.dot(xfull[:,:n_good_dims],trans_mat)
    noise = np.random.randn(y.shape[0], y.shape[1]) * 0.01
    y += noise
    targets = (y > y.mean()).astype('f')
    X = imputer.fit_transform(
                    pandas.DataFrame(xfull, 
                            index=np.arange(n_samples), 
                            columns=[chr(ord('a')+i)  for i in xrange(xfull.shape[1]) ]
                    )
                  )
    Y = pandas.Series(targets.flatten(), name='targets')
    Z = Partition( len(Y) )

    return X,Y,Z

class TestTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger('test_tasks_common')
        cls.logger.setLevel(logging.DEBUG)

        cls.logger.info('Begin new test')

        BaseTransformer = ModelingMachine.engine.tasks.transformers.BaseTransformer
        BaseModelTransformer = ModelingMachine.engine.tasks.logistic.BaseModelTransformer

        X,Y,Z = get_default_data()
        cls.bin_d_X = X
        cls.bin_d_Y = Y
        cls.bin_d_Z = Z
        

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(self):
        self.logger.info('Finishing test')

    def dev_test(self):
        """This test specifies a single task to test.

        You can run it by:
        >>> python -m unittest test_tasks_common.TestTasks.dev_test
        
        """
        test_code = ''
        if len(test_code) > 1:
            self.check_code(test_code)
        else:
            self.logger.warn('No code was specified in TaskTests.dev_test')
        

    def check_code(self, key_code):
        self.logger.info( 'Now testing code %s' % key_code )
        TaskClass = task_map[key_code]
        if issubclass(TaskClass, ModelingMachine.engine.tasks.converters.BaseConverter):
            # NOTE sensible tests for converters could be called here
            pass
        elif issubclass(TaskClass, ModelingMachine.engine.tasks.transformers.BaseTransformer):
            self.check_transformer_common(key_code)
        elif issubclass(TaskClass, ModelingMachine.engine.tasks.logistic.BaseModelTransformer):
            self.check_model_transformer_common(key_code)
        else:
            #The task should inherit from one of these three so we know which methods to test
            self.assertTrue(False, 'Task %s does not inherit from a known BaseClass' % key_code)

    def test_tasks_common(self):
        ### Put a code in this list if  you want to skip it in this test
        dev_skip_codes = [] 
        will_fail = ['GBCA', 'GBCTop', 'GBRA', 'GBRTop', 'RGAMB', 'SGDMR', 'STK', 'TML_LR1' ]
        hangs = ['DGLMA', 'DGLMPE', 'ELGMR', 'ELPR', 'ELTR', 'GBMCR', 'GBMPR', 'RGE', 'RFI']
        also_slow = ['DGLMTAE', 'EL1', 'ELGC', 'ELBC', 'GBC', 'GBCT', 'GBRT', 'GBRTR', 'GLMA', 'LR1', 'RFE', 'RFR', 'RGAM', 'RGBCT', 'RGBRA', 'RGBRP', 'RGBRT', 'SGDRH', 'SVCR', 'SVRR']

        dev_skip_codes += also_slow
        dev_skip_codes += will_fail
        dev_skip_codes += hangs

        failed_codes = []

        n_codes = len(task_map.keys())
        for key_code in sorted( task_map.keys() ):
            if key_code in dev_skip_codes: 
                continue
            try:
                self.check_code(key_code)
            except Exception as e:
                traceback.print_exc()
                self.logger.warn( 'Task %s(%s) excepted with %s' % (key_code, task_map[key_code], e) )
                failed_codes.append(key_code)
        self.assertTrue(len(failed_codes) == 0, 'The following codes failed: %s' % failed_codes)


    def check_transformer_common(self, transformer_short_code):
        """Check a Transformer class for required methods and expected outputs
        """
        TransformerClass = task_map[transformer_short_code]
        transf_object = TransformerClass()
        self.check_fit_method(transf_object)
        self.check_transform_method(transf_object)

    def check_model_transformer_common(self, model_trans_short_code):
        """Check each ModelTransformer (i.e. an estimator ) for their required methods
        """

        EstimatorClass = task_map[model_trans_short_code]
        estim_object = EstimatorClass()
        self.check_fit_method(estim_object)
        self.check_predict_method(estim_object)

    def check_fit_method(self, task_object):
        """Check that the task_object has a fit method with the right signature and that it
        returns self
        """

        fit_result = task_object.fit( X = self.bin_d_X, Y = self.bin_d_Y, Z = self.bin_d_Z )
        self.assertIs(fit_result, task_object , 'Fit must return self')
        # Check that output of fit can be pickled
        t = tempfile.TemporaryFile(mode='wb')
        cPickle.dump(fit_result, t)
        t.close()


    def check_transform_method(self, task_object):
        transf_result = task_object.transform( X = self.bin_d_X, Y = self.bin_d_Y, Z = self.bin_d_Z )
        self.assertTrue(isinstance(transf_result, ModelingMachine.engine.container.Container), 
                    'Transform method must return a container')
        for p in self.bin_d_Z:
            self.assertGreaterEqual(len(transf_result.colnames(**p)), 1, 'No colnames provided in transform result')
        # Check that the output has the same number of rows as the input
        for z in self.bin_d_Z:
            data = self.bin_d_X(**z)
            trans = transf_result(**z)
            self.assertEqual(data.shape[0], trans.shape[0])

    def check_predict_method(self, task_object):
        pred_result = task_object.predict( X = self.bin_d_X, Y = self.bin_d_Y, Z = self.bin_d_Z )
        self.assertTrue(isinstance(pred_result, ModelingMachine.engine.container.Container),
                        'Predict method must return a container')
        # Check that prediction has same number of rows as input
        for z in self.bin_d_Z:
            data = self.bin_d_X(**z)
            pred = pred_result(**z)
            self.assertEqual(data.shape[0], pred.shape[0])
            self.assertEqual(pred.shape[1], 1)  # A prediction is a single value
        #TODO - check that the output predictions is 'sensible' - depends on task



if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger('').setLevel( logging.DEBUG )
    unittest.main(verbosity=10)
