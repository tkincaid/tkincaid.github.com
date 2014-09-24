#########################################################
#
#       Unit Test for the Vertex class
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pandas
import os
import copy
import numpy as np
import pandas as pd
import logging
import pytest
from mock import patch
from ModelingMachine.engine.monitor import FakeMonitor
from ModelingMachine.engine.vertex import Vertex
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.response import Response
from ModelingMachine.engine.response import y_transform
from ModelingMachine.engine.response import pred_inv_transform

class VertexTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.vertex_monitor_patch = patch('ModelingMachine.engine.vertex.Monitor', FakeMonitor)
        self.vertex_monitor_mock = self.vertex_monitor_patch.start()

        here = os.path.dirname(os.path.abspath(__file__))
        self.ds = pandas.read_csv(os.path.join(here,'../testdata/allstate-nonzero-200.csv'))
        self.ds = self.ds.take(range(50))

    @classmethod
    def tearDownClass(cls):
        super(VertexTest, cls).tearDownClass()
        cls.vertex_monitor_patch.stop()

    def test_parse_task(self):
        vertex = Vertex(['RFR'],'id')
        self.assertEqual(vertex.tasks.parse_task('RFI 10;100;1000'),('RFI','10;100;1000'))
        self.assertEqual(vertex.tasks.parse_task('RFI 10;100 ; 1000'),('RFI','10;100;1000'))
        self.assertEqual(vertex.tasks.parse_task('RFI 10; 100;1000'),('RFI','10;100;1000'))
        self.assertEqual(vertex.tasks.parse_task('RFI    10= foo b;'),('RFI','10=foo b;'))

    def test_parse_task_whitespace(self):
        vertex = Vertex(['RFR'],'id')
        self.assertEqual(vertex.tasks.parse_task('RFI t_m=foo bar'),('RFI','t_m=foo bar'))
        self.assertEqual(vertex.tasks.parse_task('RFI    '),('RFI',''))

    def test_parse_args(self):
        vertex = Vertex(['RFR'],'id')
        self.assertEqual(vertex.parse_args('10;100;1000'),('10;100;1000',None,None))
        self.assertEqual(vertex.parse_args('logy;10;100;1000'),('10;100;1000',None,'logy'))
        self.assertEqual(vertex.parse_args('10;100;1000;logy'),('10;100;1000',None,'logy'))
        self.assertEqual(vertex.parse_args('10;logy;100;1000'),('10;100;1000',None,'logy'))

    def test_vertex_init(self):
        v = Vertex(['DM2 sc = 10; cm = 10000'], 'id')
        self.assertEqual(v.task_info[0]['arguments'], 'sc=10;cm=10000')

    def test_yTransform(self):
        y = Response.from_array(np.arange(1, 10))
        vertex = Vertex(['RFR'], 'id')
        self.assertTrue(np.all(y == y_transform(y ,None) ))
        self.assertTrue(np.all(y == y_transform(y, 'invalidargument') ))
        self.assertTrue(np.all(np.log(y + 1) == y_transform(y,'logy') ))
        ymin_desired = np.concatenate((np.array([0]).reshape(1,1), np.ones((8, 1)))).ravel()
        np.testing.assert_almost_equal(ymin_desired, y_transform(y -2, 'miny'))

    def test_pred_inv_transform(self):
        pred = np.arange(1, 10, dtype=float)
        X = Container()
        X.add(pred, colnames=['pred'], r=1, k=1)
        XT = Container()
        XT.add(np.log(pred + 1),colnames=['pred'],r=1,k=1)
        vertex = Vertex(['RFR'],'id')
        self.assertTrue(X == pred_inv_transform(X, None) )
        check = pred_inv_transform(XT,'logy')
        self.assertTrue(np.allclose(X(r=1,k=1) , check(r=1,k=1) ))

    def test_yTransform_miny(self):
        """Test that the Vertex class properly handles frequency
        modeling via the miny argument
        """
        y = Response.from_array(np.random.randint(2, 5, 12))
        rows = [(i % 2 == 0) for i in range(y.shape[0])]
        np.place(y, rows, 1)
        desired = np.array([(0 if row else 1) for row in rows])
        vertex = Vertex(['RFR'], 'id')
        actual = y_transform(y, 'miny')
        np.testing.assert_equal(actual, desired)

    def test_yTransform_miny(self):
        """Test that the Vertex class properly handles frequency
        modeling via the miny argument
        """
        y = Response.from_array(np.random.randint(2, 5, 40))
        vertex = Vertex(['RFR'], 'id')
        actual = y_transform(y, 'sev')
        np.testing.assert_equal(actual, y - 2)

    def create_data(self):
        X = copy.deepcopy(self.ds)
        Y = Response.from_array(X.pop('Claim_Amount').values)
        X = X.take(range(21,29),axis=1)
        Z = Partition(size=X.shape[0],total_size=X.shape[0]+20,folds=5,reps=5)
        Z.set(max_reps=1,max_folds=0)
        return Container(X),Y,Z

    def test_fit(self):
        X,Y,Z = self.create_data()
        tasks = ['NI','GLMG']
        vertex = Vertex(tasks,'id')
        out = vertex.fit(X,Y,Z)
        self.assertIsInstance(out,Vertex)

    def test_transform(self):
        X,Y,Z = self.create_data()
        tasks = ['NI','SGDR logy']
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        out= vertex.transform(X,Y,Z)
        self.assertIsInstance(out,Container)
        for p in out:
            a = out(**p)
            self.assertIsInstance(a,np.ndarray)
            self.assertLessEqual(a.shape[1],X.dataframe.shape[1])
            self.assertEqual(a.shape[0],X.dataframe.shape[0])
        tasks = ['NI','SGDR']
        Y = np.log(Y+1)
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        print
        print 'test transform'
        print
        out= vertex.transform(X,Y,Z,args='t=0')
        print
        print
        self.assertIsInstance(out,Container)
        for p in out:
            b = out(**p)
            self.assertIsInstance(b,np.ndarray)
            self.assertLessEqual(b.shape[1],X.dataframe.shape[1])
            self.assertEqual(b.shape[0],X.dataframe.shape[0])
        self.assertTrue( np.allclose(a,b) )

        out2= vertex.transform(X,Y,Z,args='t=.9')
        for p in out2:
            b = out(**p)
            c = out2(**p)
            self.assertLessEqual(c.shape[1],b.shape[1])
            self.assertEqual(b.shape[0],c.shape[0])

        out= vertex.transform(X,Y,Z,args='t=1;min=5')
        for p in out:
            b = out(**p)
            self.assertEqual(b.shape[1],5)
            self.assertEqual(b.shape[0],X.dataframe.shape[0])

        out= vertex.transform(X,Y,Z,args='t=0;max=4')
        for p in out:
            b = out(**p)
            self.assertEqual(b.shape[1],4)
            self.assertEqual(b.shape[0],X.dataframe.shape[0])

        out= vertex.transform(X,Y,Z,args='if=0.8')
        for p in out:
            b = out(**p)
            print b.shape
            self.assertLessEqual(b.shape[1],X.dataframe.shape[1])
            self.assertEqual(b.shape[0],X.dataframe.shape[0])


    def test_predict(self):
        X,Y,Z = self.create_data()
        tasks = ['NI','GLMG logy']
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        out= vertex.predict(X,Y,Z)
        self.assertIsInstance(out,Container)
        for p in out:
            a = out(**p)
            self.assertIsInstance(a,np.ndarray)
            self.assertLessEqual(a.shape[1],1)
            self.assertEqual(a.shape[0],X.dataframe.shape[0])

        out = vertex.predict(X, Y, Z)
        self.assertIsInstance(out,Container)
        for p in out:
            a = out(**p)
            self.assertIsInstance(a,np.ndarray)
            self.assertLessEqual(a.shape[1],1)
            self.assertEqual(a.shape[0],X.dataframe.shape[0])

        tasks = ['NI','GLMG']
        Y = np.log(Y+1)
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        out= vertex.predict(X,Y,Z)
        self.assertIsInstance(out,Container)
        for p in out:
            b = out(**p)
            self.assertIsInstance(b,np.ndarray)
            self.assertLessEqual(b.shape[1],1)
            self.assertEqual(b.shape[0],X.dataframe.shape[0])
        self.assertTrue( np.allclose(a,np.exp(b)-1) )

    def test_stack(self):
        X,Y,Z = self.create_data()
        tasks = ['NI','SGDR logy']
        inputs = set(['N'])
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        out= vertex.stack(X,Y,Z,inputs=inputs)
        self.assertIsInstance(out,Container)
        for p in out:
            a = out(**p)
            self.assertIsInstance(a,np.ndarray)
            self.assertLessEqual(a.shape[1],1)
            self.assertEqual(a.shape[0],X.dataframe.shape[0])
        tasks = ['NI','SGDR']
        Y = np.log(Y+1)
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        out= vertex.stack(X,Y,Z,inputs=inputs)
        self.assertIsInstance(out,Container)
        for p in out:
            b = out(**p)
            self.assertIsInstance(b,np.ndarray)
            self.assertLessEqual(b.shape[1],1)
            self.assertEqual(b.shape[0],X.dataframe.shape[0])
        self.assertTrue( np.allclose(a,np.exp(b)-1) )

    def test_transform2(self):
        X,Y,Z = self.create_data()
        tasks = ['NI','RFR logy;50;5']
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        out= vertex.transform2(X,Y,Z)
        self.assertIsInstance(out,Container)
        for p in out:
            a = out(**p)
            self.assertIsInstance(a,np.ndarray)
            self.assertLessEqual(a.shape[1],X.dataframe.shape[1])
            self.assertEqual(a.shape[0],X.dataframe.shape[0])
        tasks = ['NI','RFR 50;5']
        Y = np.log(Y+1)
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        out= vertex.transform2(X,Y,Z)
        self.assertIsInstance(out,Container)
        for p in out:
            b = out(**p)
            self.assertIsInstance(b,np.ndarray)
            self.assertLessEqual(b.shape[1],X.dataframe.shape[1])
            self.assertEqual(b.shape[0],X.dataframe.shape[0])
        self.assertTrue( np.allclose(a,b) )

    def test_dump_and_load(self):
        X,Y,Z = self.create_data()
        tasks = ['NI','RFR logy;50;5']
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        for p in Z:
            load = vertex.dump((p['r'],p['k']))
        Z.set(partitions=[[1,-1]])
        vertex = Vertex(tasks,'id')
        vertex.fit(X,Y,Z)
        for p in Z:
            load2 = vertex.dump((p['r'],p['k']))
        Z.set(partitions=[[1,-1]])
        vertex = Vertex(tasks,'id')
        vertex.update(load)
        vertex.fit(X,Y,Z)
        self.assertEqual(vertex.fit_parts, set([(0,-1),(1,-1)]))
        self.assertEqual(vertex.loaded_parts, set([(0,-1)]))
        self.assertEqual(vertex.dirty_parts, set([(1,-1)]))

        test_tasks = []
        test_tasks.append( ['NI','RFR logy;50;5','ST','GLMG logy'] )
        test_tasks.append( ['GS','GLMG logy'] )
        for tasks in test_tasks:
            vertex = Vertex(tasks,'id')
            Z.set(partitions=[[0,-1],[1,-1]])
            vertex.fit(X,Y,Z)
            out = vertex.predict(X,Y,Z)
            dumps = []
            for p in Z:
                load = vertex.dump((p['r'],p['k']))
                dumps.append(load)

            vertex = Vertex(tasks,'id')
            for i,p in enumerate(Z):
                vertex.update(dumps[i])
            out2 = vertex.predict(X,Y,Z)

            self.assertEqual(len(list(Z)),2)
            self.assertEqual(list(out),list(Z))
            self.assertEqual(list(out),list(out2))
            for p in out:
                self.assertTrue(np.all(out(**p)==out2(**p)))

    def test_load_priority(self):
        """ when loading data from multiple partitions, the attribute value from 
        first partition should be used in case of conflicting values
        """
        X,Y,Z = self.create_data()
        Z.set(partitions=[[0,-1]])
        tasks = ['NI']
        vertex1 = Vertex(tasks,'id')
        vertex1.fit(X,Y,Z)
        vertex1.steps[0][0].testattribute = 1
        store1 = vertex1.save()

        Z.set(partitions=[[1,-1]])
        vertex2 = Vertex(tasks,'id')
        vertex2.fit(X,Y,Z)
        vertex2.steps[0][0].testattribute = 2
        store2 = vertex2.save()

        store2.update(store1)
        vertex3 = Vertex(tasks,'id', stored_files=store2)
        print vertex3.steps[0][0].testattribute
        self.assertEqual(vertex3.steps[0][0].testattribute, 1)

    def test_xTransform_func_none(self):
        """Test that the xTransform returns identity if None is passed
        """
        Xdata = np.random.random(10)
        Xcont = Container()
        Xcont.add(Xdata)
        vertex = Vertex(['RFR'], 'id')
        out = vertex.xTransform(Xcont, func=None)
        self.assertEqual(out, Xcont)

    def test_xTransform_func_logx(self):
        """Test that the Xtranform correctly handles logx
        """
        Xdata = np.arange(1, 10, dtype=float)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdata, colnames=["pred"], r=1, k=1)
        Xcontsf.add(Xdata, colnames=["pred"])
        vertex = Vertex(['RFR'], 'id')
        check = np.log(Xdata).reshape(-1, 1)
        outdf = vertex.xTransform(Xcontdf, "logx")
        outsf = vertex.xTransform(Xcontsf, "logx")
        np.testing.assert_almost_equal(check, outdf(r=1, k=1))
        np.testing.assert_almost_equal(check, outsf())

    def test_xTransform_func_log1x(self):
        """Test that the Xtranform correctly handles log1+x
        """
        Xdata = np.arange(1, 10, dtype=float)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdata, colnames=["pred"], r=1, k=1)
        Xcontsf.add(Xdata, colnames=["pred"])
        vertex = Vertex(['RFR'], 'id')
        check = np.log(Xdata + 1).reshape(-1, 1)
        outdf = vertex.xTransform(Xcontdf, "log1+x")
        outsf = vertex.xTransform(Xcontsf, "log1+x")
        np.testing.assert_almost_equal(check, outdf(r=1, k=1))
        np.testing.assert_almost_equal(check, outsf())

    def test_xTransform_func_logitx(self):
        """Test that the xTransform correctly handles logitx
        """
        Xdata = np.arange(1, 10, dtype=float) / 10
        Xcontsf = Container()
        Xcontdf = Container()
        Xcontdf.add(Xdata, colnames=["pred"], r=1, k=1)
        Xcontsf.add(Xdata, colnames=["pred"])
        vertex = Vertex(['RFR'], 'id')
        X1 = np.minimum(np.maximum(0.001, Xdata), 0.999)
        check = np.log(X1 / (1 - X1)).reshape(-1, 1)
        outdf = vertex.xTransform(Xcontdf, "logitx")
        outsf = vertex.xTransform(Xcontsf, "logitx")
        np.testing.assert_almost_equal(check, outdf(r=1, k=1))
        np.testing.assert_almost_equal(check, outsf())

    def test_XTransform_sev_sets_mask(self):
        """Test that the severity transform properly sets
        the mask
        """
        xdata = np.random.randn(200).reshape((20, -1))
        ydata = Response.from_array(np.random.randint(0, 2, size=20))
        xcont = Container()
        xcont.add(xdata)
        Z = Partition(20, total_size=24,folds=5, reps=1)
        Z.set(max_reps=1, max_folds=5)
        vertex = Vertex(['RFR'], 'id')
        for p in Z:
            xcont.add(xdata, **p)
        outcont = vertex.xTransform(xcont, "sev", ydata)
        for p in Z:
            outmask = outcont.get_mask(**p)
            print p
            print outmask
            # Mask should be boolean
            self.assertEqual(outmask.dtype.kind, 'b')
            desiredmask = ydata > 0
            np.testing.assert_array_equal(desiredmask, outmask)

    def test_XTransform_sev_fit_sets_mask(self):
        """Test that severity transform runs successfully and
        that after Vertex.fit() it contains the correct mask
        """
        xdata = np.random.randn(400).reshape((40, -1))
        ydata = Response.from_array(np.random.randint(0, 2, size=40).reshape(-1, 1))
        xcont = Container()
        xcont.add(xdata)
        Z = Partition(40, total_size=24,folds=5, reps=1)
        Z.set(max_reps=1, max_folds=5)
        vertex = Vertex(['RFR sev'], 'id')
        vertex.fit(xcont, ydata, Z)
        for p in xcont:
            self.assertEqual(xcont.get_mask(**p), ydata > 0)

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    unittest.main()
