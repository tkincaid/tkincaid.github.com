#########################################################
#
#       Unit Test for tasks/converters.py
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import logging
import pandas as pd
import numpy as np

from base_task_test import BaseTaskTest
from ModelingMachine.engine.tasks.converters import BaseConverter
from ModelingMachine.engine.tasks.converters import Constant_splines
from ModelingMachine.engine.tasks.converters import SingleColumnText
from ModelingMachine.engine.container import Container

class TestConverters(BaseTaskTest):

    def test_sample(self):
        con = BaseConverter()
        X,Y,Z = self.create_bin_data()
        Z.set(samplepct=8)
        X = X.dataframe
        out = con._sample(X,Z)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(out.shape,(16,X.shape[1]))
        self.assertEqual(list(X.columns),list(out.columns))

        out = con._sample(X)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(out.shape,X.shape)
        self.assertEqual(list(X.columns),list(out.columns))

    def test_constant_splines(self):
        """Regression test to make sure that constant splines return same output as before. """
        reference = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        rng = np.random.RandomState(13)
        X = rng.rand(100, 6)
        X = pd.DataFrame(data=X, columns=list('abcdef'))
        cs = Constant_splines()
        cs.fit(Container(X))
        T = cs.transform(Container(X.ix[:10]))
        res = T().tolist()
        self.assertEqual(res, reference)


class TestSingleColumnText(BaseTaskTest):
    def test_singleColumnText(self):
        col1 = ['Lorem ipsum dolor', 'sit amet,', 'consectetur adipiscing elit.', 'Donec a diam lectus', 'Sed sit amet ipsum mauris']
        col2 = ['Vivamus fermentum semper porta', 'Nunc diam velitm adipiscing ut', 'trisitque vitae', 'saggittis vel odio.', 'Macenas convallis ullamcorper ultricies.']
        # Feel free to choose a different set of unicode characters,
        col3 = [unichr(1344), unichr(8961), unichr(438), unichr(97), unichr(10)]
        df = pd.DataFrame({'lorem1': col1, 'lorem2': col2, 'unicode1': col3})
        df= Container(df)
        task1 = SingleColumnText('cn=0')
        task2 = SingleColumnText('cn=1')
        task3 = SingleColumnText('cn=2')
        task1.fit(df)
        out1 = task1.transform(df)
        self.assertEqual(out1().shape, (5, 1))
        self.assertEqual(out1.colnames(), ['lorem1'])
        task2.fit(df)
        out2 = task2.transform(df)
        self.assertEqual(out2().shape, (5, 1))
        self.assertEqual(out2.colnames(), ['lorem2'])
        task3.fit(df)
        out3 = task3.transform(df)
        self.assertEqual(out3().shape, (5, 1))
        self.assertEqual(out3.colnames(), ['unicode1'])


if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
