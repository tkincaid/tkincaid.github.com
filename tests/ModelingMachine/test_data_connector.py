import unittest
import pandas
import numpy as np
from rpy2 import robjects
from ModelingMachine.engine.data_connector import DataConnector
from ModelingMachine.engine.container import Container

class DataConnectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dc = DataConnector()
        cls.pandas = pandas.DataFrame({'a':[1,2,3],'b':[3,4,5],'c':[4,5,6]})
        cls.R = robjects.r('data.frame(a = c(1,2,3),b = c(3,4,5),c = c(4,5,6))')
        cls.container = Container()
        cls.container.add(np.array([[1,3,4],[2,4,5],[3,5,6]]),colnames=['a','b','c'])
        #cls.container.add(np.array([(1,3,'a'),(2,4,'b'),(3,5,'c')],dtype=[('a','<i4'),('b','<i4'),('c','|S1')]),colnames=['a','b','c'])

    def test_R_to_Pandas(self):
        d = self.dc.get(self.R, out_fmt=DataConnector.PANDAS)
        self.assertTrue(np.all(d.values == self.pandas.values))
        self.assertTrue(np.all(d.columns == self.pandas.columns))

    def test_R_to_Container(self):
        d = self.dc.get(self.R, out_fmt=DataConnector.CONTAINER)
        self.assertTrue(np.all(d() == self.container()))
        self.assertTrue(np.all(d.colnames() == self.container.colnames()))

    def test_Pandas_to_R(self):
        d = self.dc.get(self.pandas, out_fmt=DataConnector.R)
        robjects.globalenv['d1'] = d
        robjects.globalenv['d2'] = self.R
        self.assertTrue(np.all(robjects.r('d1==d1')))

    def test_Pandas_to_Container(self):
        d = self.dc.get(self.pandas, out_fmt=DataConnector.CONTAINER)
        self.assertTrue(np.all(d() == self.container()))
        self.assertTrue(np.all(d.colnames() == self.container.colnames()))

    def test_Container_to_R(self):
        d = self.dc.get(self.container, out_fmt=DataConnector.R)
        robjects.globalenv['d1'] = d
        robjects.globalenv['d2'] = self.R
        self.assertTrue(np.all(robjects.r('d1==d1')))

    def test_Container_to_Pandas(self):
        d = self.dc.get(self.container, out_fmt=DataConnector.PANDAS)
        self.assertTrue(np.all(d.values == self.pandas.values))
        self.assertTrue(np.all(d.columns == self.pandas.columns))

if __name__ == '__main__':
    unittest.main()
