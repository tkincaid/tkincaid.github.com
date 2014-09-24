#########################################################
#
#       Unit Test for GLM extension module
#
#       Author: Glen Koundry
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest
import logging
import json
import cPickle
import numpy as np
from ModelingMachine.engine.tasks.GLM import GLM


class TestGLMModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.load(open('tests/testdata/fixtures/glm_regression.json', 'r'))
        np.random.seed(1234)
        cls.offsets = np.minimum(12, np.random.randint(1, 20, 200)).astype(float)
        cls.weights = np.random.randint(1, 5, 200).astype(float)

    def regression_test(self, X, y, dist, offsets, weights, array_fmt='C', exp_key=None):
        model = GLM()
        args = {}
        if offsets is not None:
            args['offsets'] = offsets[:100]
        if weights is not None:
            args['weights'] = weights[:100]
        if array_fmt=='C':
            train = np.ascontiguousarray(X[:100])
            test = np.ascontiguousarray(X[100:])
        else:
            train = np.asfortranarray(X[:100])
            test = np.asfortranarray(X[100:])
        model.fit(train.astype(float),
                  y[:100].astype(float),
                  distribution=dist,
                  p=1.2, tweedie_log=True, **args)
        args = {}
        if offsets is not None:
            args['offsets'] = offsets[100:]
        pred = model.predict(test.astype(float),**args)
        #import sys
        #sys.stderr.write('"%s,%s,%s": %s,\n' % (dist, offsets is not None, weights is not None, pred.tolist()))
        if exp_key is None:
            exp_key = '%s,%s,%s' % (dist, offsets is not None, weights is not None)
        np.testing.assert_array_almost_equal(self.expected[exp_key], pred)

        # check that pickling works
        pickle = cPickle.dumps(model)
        model = cPickle.loads(pickle)
        pred = model.predict(X[100:].astype(float),**args)
        np.testing.assert_array_almost_equal(self.expected[exp_key], pred)

    def test_regression_gaussian(self):
        np.random.seed(1234)
        y = np.random.random(200)*10
        X = np.array([
            y**2,
            np.abs(y-5),
            np.random.random(200)
        ]).T
        self.regression_test(X, y, "Gaussian", None, None)
        self.regression_test(X, y, "Gaussian", self.offsets, None)
        self.regression_test(X, y, "Gaussian", None, self.weights)
        self.regression_test(X, y, "Gaussian", self.offsets, self.weights)

    def test_regression_poisson(self):
        np.random.seed(1234)
        y = np.random.poisson(10, 200)
        X = np.array([
            y**2,
            np.abs(y-5),
            np.random.random(200)
        ]).T
        self.regression_test(X, y, "Poisson", None, None)
        self.regression_test(X, y, "Poisson", self.offsets, None)
        self.regression_test(X, y, "Poisson", None, self.weights)
        self.regression_test(X, y, "Poisson", self.offsets, self.weights)

    def test_regression_gamma(self):
        np.random.seed(1234)
        y = np.random.gamma(10, 20, 200)
        X = np.array([
            y**2,
            np.abs(y-5),
            np.random.random(200)
        ]).T
        self.regression_test(X, y, "Gamma", None, None)
        self.regression_test(X, y, "Gamma", self.offsets, None)
        self.regression_test(X, y, "Gamma", None, self.weights)
        self.regression_test(X, y, "Gamma", self.offsets, self.weights)

    def test_regression_tweedie(self):
        np.random.seed(1234)
        y = np.random.gamma(2, 2, 200) * np.random.poisson(2, 200)
        X = np.array([
            y**2,
            np.abs(y-5),
            np.random.random(200)
        ]).T
        self.regression_test(X, y, "Tweedie", None, None)
        self.regression_test(X, y, "Tweedie", self.offsets, None)
        self.regression_test(X, y, "Tweedie", None, self.weights)
        self.regression_test(X, y, "Tweedie", self.offsets, self.weights)

    def test_regression_bernoulli(self):
        np.random.seed(1234)
        y = np.random.binomial(1, 0.3, 200)
        X = np.array([
            np.random.randint(0, 2, 200) + y,
            np.random.random(200) * (y + 1),
            np.random.random(200)
        ]).T
        self.regression_test(X, y, "Bernoulli", None, None)
        self.regression_test(X, y, "Bernoulli", self.offsets, None)
        self.regression_test(X, y, "Bernoulli", None, self.weights)
        self.regression_test(X, y, "Bernoulli", self.offsets, self.weights)

    def test_array_format(self):
        np.random.seed(1234)
        y = np.random.random(200)*10
        X = np.array([
            y**2,
            np.abs(y-5),
            np.random.random(200)
        ]).T
        # test C format
        self.regression_test(np.ascontiguousarray(X), y, "Gaussian", None, None, array_fmt='C')
        # test fortran format
        self.regression_test(np.ascontiguousarray(X), y, "Gaussian", None, None, array_fmt='fortran')

    def test_singular_matrix(self):
        np.random.seed(1234)
        y = np.random.random(200)*10
        X = np.array([
            np.arange(200),
            np.arange(200),
            np.arange(200),
        ]).T
        self.regression_test(np.ascontiguousarray(X), y, "Gaussian", None, None, exp_key='singular')

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
