######################################################################
#
#       unit test for Container
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
######################################################################

import unittest
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
import operator

#mm = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(mm,'../..'))

from ModelingMachine.engine.container import Container, _ImmutableDictView
from ModelingMachine.engine.container import safe_hstack, _deep_update
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.vertex import check_output_weights
from copy import deepcopy


class DummyClass():
    shape = [1]

dummy = DummyClass()


class TestContainerFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cols1 = ['A','B','C']
        cls.cols2 = ['D','E','F']
        cls.types1 = [0,0,0]
        cls.types2 = [3,4,5]
        cls.tmp = np.array([[1,2,3],[4,5,6],[7,8,9]])
        cls.tmp1 = sp.coo_matrix([[1,0,0],[0,1,0],[0,0,1]])
        cls.tmp2 = np.array([[1],[1],[1]])
        cls.test_cols = ['C1','C2','C3']
        cls.cont = Container()
        cls.cont.add(cls.tmp,colnames=cls.test_cols)
        cls.cont.add(cls.tmp1)
        cls.test_cols2 = cls.test_cols+cls.test_cols
        cls.test_cols += ['_ss0','_ss1','_ss2']
        cls.part = Partition(1000,total_size=1200)
        cls.part.set(max_reps=1)
        cls.part_list = []
        cls.test_list = []
        cls.test_list2 = []
        for i in cls.part:
            tmp3 = cls.tmp2.copy()
            cls.cont.add(tmp3,**i)
            cls.tmp2 += cls.tmp2
            cls.part_list.append(i)
            cls.test_list.append(np.concatenate([cls.tmp,cls.tmp1.toarray(),tmp3],axis=1))
            cls.test_list2.append(np.concatenate([cls.tmp,cls.tmp,cls.tmp1.toarray(),cls.tmp1.toarray(),tmp3,tmp3],axis=1))
        cls.test_cols += ['_da0']
        cls.tc = Container()
        cls.tc.add(cls.tmp,colnames=cls.cols1)
        cls.tc_cat = Container()
        cls.tc_cat.add(cls.tmp,colnames=cls.cols1,coltypes=cls.types2)
        cls.tc_num_cat=cls.tc+cls.tc_cat
        cls.tc_cat2=cls.tc_cat+cls.tc_cat
        cls.tc2 = Container()
        cls.tc2.add(cls.tmp1, colnames=cls.cols2)
        cls.tc3 = cls.tc+cls.tc2
        cls.tc4 = cls.tc+cls.tc
        cls.test_list3 = []
        cls.tc5=deepcopy(cls.tc3)
        for i in cls.part:
            tmp3 = cls.tmp2.copy()
            i['colnames'] = ['P']
            cls.tc5.add(tmp3,**i)
            cls.test_list3.append(tmp3)

    def test_iteration(self):
        for n,i in enumerate(self.cont):
            self.assertEqual( i , self.part_list[n])
            self.assertTrue( np.all(self.cont.colnames(**i) == self.test_cols))
            self.assertTrue( np.all(self.cont.coltypes(**i) == self.types1+self.types1+[0]))
            self.assertTrue( np.all(self.tc_num_cat.coltypes(**i) == self.types1+self.types2))

    def test_add(self):
        cont2 = self.cont + self.cont
        cont=deepcopy(self.cont)
        cont += cont
        self.assertTrue( np.all(cont().toarray() == cont2().toarray()))
        self.assertListEqual( cont.keys(),cont2.keys())
        test_cols2 = ['C1','C2','C3','C1','C2','C3','_ss0','_ss1','_ss2','_ss0','_ss1','_ss2','_da0','_da0']
        for n,i in enumerate(cont):
            self.assertTrue(np.all(cont.colnames(**i) == test_cols2))
            self.assertTrue(np.all( cont(**i).toarray() == cont2(**i).toarray() ))
        self.assertEqual(len(cont),5)
        with self.assertRaises(TypeError):
            cont._combine(np.array([1,2,3]),5)
        c1=Container()
        for i in self.part:
            with self.assertRaises(ValueError):
                cont.add(dummy)
            with self.assertRaises(ValueError):
                cont.add(dummy,**i)
            c1.add(self.tmp1,**i)
            c1.add(self.tmp,**i)
            c1.add(self.tmp1,**i)
            expected = sp.hstack( (sp.csr_matrix(self.tmp),sp.csr_matrix(self.tmp1),sp.csr_matrix(self.tmp1)), format='csr')
            self.assertTrue(sp.issparse(c1(**i)))
            self.assertTrue(np.all(c1(**i).toarray()==expected.toarray()))


    def test_dynamic_only(self):
        test_cols2 = ['_da0',]
        for n,i in enumerate(self.cont):
            i['dynamic_only']=True
            self.assertTrue(np.all(self.cont(**i) == [[j[12],] for j in self.test_list2[n]]))
            self.assertTrue(np.all(self.cont.colnames(**i) == test_cols2))

    def test_set_type(self):
        self.cont._set_type()

    def test_1d(self):
        a = np.array(range(10))
        b = np.array(range(10,20))
        cont1 = Container()
        cont2 = Container()
        cont1.add(a)
        cont2.add(b)
        self.assertTrue(np.all(cont1()==np.column_stack([a])))
        self.assertListEqual(cont1.colnames(),['_sa0'])
        self.assertTrue(np.all(cont2()==np.column_stack([b])))
        self.assertListEqual(cont2.colnames(),['_sa0'])
        x = cont1 + cont2
        self.assertListEqual(x.colnames(),['_sa0','_sa0'])
        cont1 += cont2
        self.assertTrue(np.all(cont1()==x()))
        self.assertListEqual(cont1.colnames(),['_sa0','_sa0'])

    def test_colnames(self):
        self.assertTrue(np.all(self.tc()==self.tmp))
        self.assertTrue(np.all(self.tc.colnames()==self.cols1))
        self.assertTrue(np.all(self.tc2().toarray()==self.tmp1))
        self.assertTrue(np.all(self.tc2.colnames()==self.cols2))
        self.assertTrue(np.all(self.tc3.colnames()==self.cols1+self.cols2))
        self.assertTrue(np.all(self.tc4.colnames()==self.cols1+self.cols1))
        tc=deepcopy(self.tc)
        tc+=tc
        self.assertTrue(np.all(tc()==self.tc4()))
        self.assertTrue(np.all(tc.colnames()==self.tc4.colnames()))

    def test_coltypes(self):
        self.assertTrue(np.all(self.tc.coltypes()==self.types1))
        self.assertTrue(np.all(self.tc_cat.coltypes()==self.types2))
        self.assertTrue(np.all(self.tc_num_cat.coltypes()==self.types1+self.types2))
        tc=deepcopy(self.tc)
        tc+=tc
        self.assertTrue(np.all(tc.coltypes()==self.tc4.coltypes()))
        tc_cat=deepcopy(self.tc_cat)
        tc_cat+=tc_cat
        self.assertTrue(np.all(tc_cat.coltypes()==self.tc_cat2.coltypes()))

    def test_named_feature(self):
        for n,i in enumerate(self.tc5):
            self.assertEqual(i,self.part_list[n])
            self.assertTrue(np.all(self.tc5.colnames(**i)==self.cols1+self.cols2+['P']))

    def test_unnamed_features(self):
        tc3=deepcopy(self.tc5)
        for i in self.part:
            tc3.add(self.tmp1,**i)
        self.assertTrue(np.all(tc3.colnames()==self.cols1+self.cols2))
        for n,i in enumerate(tc3):
            self.assertEqual(i,self.part_list[n])
            self.assertTrue(np.all(tc3.colnames(**i)==self.cols1+self.cols2+['P','_ds0','_ds1','_ds2']))

    def test_combine(self):
        c = Container()
        x,s = deepcopy(self.tmp), deepcopy(self.tmp1)
        out = c._combine(x,x)
        expected = np.hstack((x,x))
        self.assertEqual(sp.issparse(out),False)
        self.assertEqual(np.all(out==expected),True)
        out = c._combine(s,s)
        expected = sp.hstack((s,s),format='csr')
        self.assertEqual(sp.issparse(out),True)
        self.assertEqual(np.all(out.toarray()==expected.toarray()),True)
        out = c._combine(x,s)
        expected = sp.hstack((sp.csr_matrix(x),s),format='csr')
        self.assertEqual(sp.issparse(out),True)
        self.assertEqual(np.all(out.toarray()==expected.toarray()),True)
        out = c._combine(s,x)
        expected = sp.hstack((s,sp.csr_matrix(x)),format='csr')
        self.assertEqual(sp.issparse(out),True)
        self.assertEqual(np.all(out.toarray()==expected.toarray()),True)

    def test_transform_container_func_logx(self):
        """Test that the tranform_containe correctly applies logx
        """
        Xdata = np.arange(1, 10)
        check = np.log(Xdata).reshape(-1, 1)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdata, colnames=["pred"], r=1, k=1)
        Xcontdf.transform_container('logx')
        Xcontsf.add(Xdata, colnames=["pred"])
        Xcontsf.transform_container('logx')
        np.testing.assert_almost_equal(check, Xcontdf(r=1, k=1))
        np.testing.assert_almost_equal(check, Xcontsf())

    def test_transform_container_func_log1x(self):
        """Test that the tranform_containe correctly applies log1+x
        """
        Xdata = np.arange(1, 10)
        check = np.log(Xdata + 1).reshape(-1, 1)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdata, colnames=["pred"], r=1, k=1)
        Xcontdf.transform_container('log1+x')
        Xcontsf.add(Xdata, colnames=["pred"])
        Xcontsf.transform_container('log1+x')
        np.testing.assert_almost_equal(check, Xcontdf(r=1, k=1))
        np.testing.assert_almost_equal(check, Xcontsf())

    # Sparse matrices not supported yet
    @pytest.mark.skip
    def xtest_transform_container_func_logx_sparse(self):
        """Test that the xTransform correctly applies logx
        if the data is sparse
        """
        Xdata = np.arange(1, 10, dtype=float) / 10
        check = np.log(Xdata).reshape(-1, 1)
        Xdatasparse = sp.coo_matrix(Xdata)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdatasparse, colnames=["pred"], r=1, k=1)
        Xcontdf.transform_container('logx')
        Xcontsf.add(Xdatasparse, colnames=["pred"])
        Xcontsf.transform_container('logx')
        np.testing.assert_almost_equal(check, Xcontdf(r=1, k=1).todense())
        np.testing.assert_almost_equal(check, Xcontsf().todense())

    @pytest.mark.skip
    def xtest_transform_container_func_log1x_sparse(self):
        """Test that the xTransform correctly applies log1+x
        if the data is sparse
        """
        Xdata = np.arange(1, 10, dtype=float) / 10
        check = np.log(Xdata + 1).reshape(-1, 1)
        Xdatasparse = sp.coo_matrix(Xdata)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdatasparse, colnames=["pred"], r=1, k=1)
        Xcontdf.transform_container('log1+x')
        Xcontsf.add(Xdatasparse, colnames=["pred"])
        Xcontsf.transform_container('log1+x')
        np.testing.assert_almost_equal(check, Xcontdf(r=1, k=1).todense())
        np.testing.assert_almost_equal(check, Xcontsf().todense())

    def test_transform_container_func_logitx(self):
        """Test that the xTransform correctly applies logitx
        """
        Xdata = np.arange(1, 10, dtype=float) / 10
        X1 = np.minimum(np.maximum(0.001, Xdata), 0.999)
        check = np.log(X1 / (1 - X1)).reshape(-1, 1)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdata, colnames=["pred"], r=1, k=1)
        Xcontdf.transform_container('logitx')
        Xcontsf.add(Xdata, colnames=["pred"])
        Xcontsf.transform_container('logitx')
        np.testing.assert_almost_equal(check, Xcontdf(r=1, k=1))
        np.testing.assert_almost_equal(check, Xcontsf())

    # Sparse matrices not supported yet
    @pytest.mark.skip
    def xtest_transform_container_func_logitx_sparse(self):
        """Test that the xTransform correctly applies logitx
        if the data is sparse
        """
        Xdata = np.arange(1, 10, dtype=float) / 10
        X1 = np.minimum(np.maximum(0.001, Xdata), 0.999)
        check = np.log(X1 / (1 - X1)).reshape(-1, 1)
        Xdatasparse = sp.csc_matrix(Xdata)
        Xcontdf = Container()
        Xcontsf = Container()
        Xcontdf.add(Xdatasparse, colnames=["pred"], r=1, k=1)
        Xcontdf.transform_container('logitx')
        Xcontsf.add(Xdatasparse, colnames=["pred"])
        Xcontsf.transform_container('logitx')
        np.testing.assert_almost_equal(check, Xcontdf(r=1, k=1).todense())
        np.testing.assert_almost_equal(check, Xcontsf().todense())

    def test_set_mask_all_partitions(self):
        """Test that the set_mask function correctly works when
        applied to all partitions
        """
        rkeys = range(0, 5)
        kkeys = range(1, 5)
        cont = Container()
        for r in rkeys:
            for k in kkeys:
                Xdata = np.random.randn(50).reshape(10, -1)
                cont.add(Xdata, r=str(r), k=str(k))
        mask = [False, False, False, False, False,
                True, True, True, True, True]
        cont.set_mask(mask)
        out = cont.get_mask()
        np.testing.assert_array_equal(out, mask)
        for r in rkeys:
            for k in kkeys:
                out = cont.get_mask(r=str(r), k=str(k))
                np.testing.assert_array_equal(out, mask)

    def test_set_mask_single_partition(self):
        """Test that the set_mask function correctly works when
        applied to a specific partition
        """
        rkeys = range(0, 5)
        kkeys = range(0, 5)
        cont = Container()
        for r in rkeys:
            for k in kkeys:
                Xdata = np.random.randn(50).reshape(10, -1)
                cont.add(Xdata, r=str(r), k=str(k))
        mask = np.array([False, False, False, False, False,
                True, True, True, True, True])
        cont.set_mask(mask, r='2', k='3')
        for r in rkeys:
            for k in kkeys:
                out = cont.get_mask(r=str(r), k=str(k))
                if r != 2 or k !=3:
                    self.assertTrue(out is None)
                else:
                    np.testing.assert_array_equal(out, mask)

    def test_remove_masks_all_partitions(self):
        """Test that when combining containers, they remove masks
        """
        rkeys = range(0, 5)
        kkeys = range(0, 5)
        cont = Container()
        for r in rkeys:
            for k in kkeys:
                Xdata = np.random.randn(50).reshape(10, -1)
                cont.add(Xdata, r=str(r), k=str(k))
        mask = [False, False, False, False, False,
                True, True, True, True, True]
        cont.set_mask(mask)
        cont.remove_mask()
        for r in rkeys:
            for k in kkeys:
                out = cont.get_mask(r=str(r), k=str(k))
                self.assertTrue(out is None)

    def test_remove_masks_single_paritions(self):
        """Test that when combining containers, they remove masks
        """
        rkeys = range(0, 5)
        kkeys = [-1]
        cont = Container()
        for r in rkeys:
            for k in kkeys:
                Xdata = np.random.randn(50).reshape(10, -1)
                cont.add(Xdata, r=str(r), k=str(k))
        mask = [False, False, False, False, False,
                True, True, True, True, True]
        cont.set_mask(mask)
        cont.remove_mask(r='1', k='-1')
        for r in rkeys:
            for k in kkeys:
                out = cont.get_mask(r=str(r), k=str(k))
                if r == 1 and k == -1:
                    self.assertTrue(out is None)
                else:
                    np.testing.assert_array_equal(out, mask)

    def test_container_add_removes_masks_all_partitions(self):
        """Test that when combining containers, they remove masks
        """
        rkeys = range(0, 5)
        kkeys = range(0, 5)
        cont1 = Container()
        cont2 = Container()
        for r in rkeys:
            for k in kkeys:
                Xdata = np.random.randn(50).reshape(10, -1)
                cont1.add(Xdata, r=str(r), k=str(k))
                Xdata2 = np.random.randn(50).reshape(10, -1)
                cont2.add(Xdata2, r=str(r), k=str(k))
        mask = [False, False, False, False, False,
                True, True, True, True, True]
        mask2 = [int(i) for i in mask]
        cont1.set_mask(mask2)
        cont3 = cont1 + cont2
        for r in rkeys:
            for k in kkeys:
                out = cont3.get_mask(r=str(r), k=str(k))
                self.assertTrue(out is None)
                out1 = cont1.get_mask(r=str(r), k=str(k))
                self.assertTrue(out1 is not None)
                out2 = cont2.get_mask(r=str(r), k=str(k))
                self.assertTrue(out2 is None)
        out = cont3.get_mask()
        self.assertTrue(out is None)
        cont2.set_mask(mask)
        cont3 = cont1 + cont2
        for r in rkeys:
            for k in kkeys:
                out = cont3.get_mask(r=str(r), k=str(k))
                self.assertTrue(out is None)
                out1 = cont1.get_mask(r=str(r), k=str(k))
                self.assertTrue(out1 is not None)
                out2 = cont2.get_mask(r=str(r), k=str(k))
                self.assertTrue(out2 is not None)
        out = cont3.get_mask()
        self.assertTrue(out is None)

    def test_container_add_equals_removes_masks_all_partitions(self):
        """Test that when combining containers, they remove masks
        """
        rkeys = range(0, 5)
        kkeys = range(0, 5)
        cont1 = Container()
        cont2 = Container()
        for r in rkeys:
            for k in kkeys:
                Xdata = np.random.randn(50).reshape(10, -1)
                cont1.add(Xdata, r=str(r), k=str(k))
                Xdata2 = np.random.randn(50).reshape(10, -1)
                cont2.add(Xdata2, r=str(r), k=str(k))
        mask = [False, False, False, False, False,
                True, True, True, True, True]
        mask2 = np.array(mask).astype(int)
        cont1.set_mask(mask2)
        cont2 += cont1
        for r in rkeys:
            for k in kkeys:
                out = cont2.get_mask(r=str(r), k=str(k))
                self.assertTrue(out is None)
                out1 = cont1.get_mask(r=str(r), k=str(k))
                self.assertTrue(out1 is not None)
        out = cont2.get_mask()
        self.assertTrue(out is None)

    def test_combine_empty_container(self):
        """Test if combination of empty containers works"""
        a = Container()
        b = Container()
        c = a + b
        self.assertEqual(c.nrow, a.nrow)
        self.assertEqual(c.nrow, b.nrow)
        self.assertEqual(c.nrow, 0)

        np.testing.assert_array_equal(a(), c())
        np.testing.assert_array_equal(b(), c())

    def test_empty_add(self):
        """Test if adding empty matrices works"""
        a = Container()
        arr = np.empty((10, 0))
        a.add(arr)
        np.testing.assert_array_equal(a(), arr)

        a = Container()
        arr = np.empty((10, 0))
        a.add(arr, r=0, k=2)
        np.testing.assert_array_equal(a(r=0, k=2), arr)

    def test_malformed_input(self):
        """Test input checks (coltypes, colnames)"""
        a = Container()
        arr = np.empty((10, 0))
        self.assertRaises(ValueError, a.add, arr, colnames=['foo'], coltypes=None)
        self.assertRaises(ValueError, a.add, arr, colnames=['foo'], coltypes=[])
        self.assertRaises(ValueError, a.add, arr, colnames=['foo'], coltypes=[0])
        self.assertRaises(ValueError, a.add, arr, colnames=None, coltypes=[0])
        self.assertRaises(ValueError, a.add, arr, colnames=[], coltypes=[0])

    def test_malformed_input_shape(self):
        """Test input checks (X.shape)"""
        a = Container()
        arr = np.empty((10, 1, 1))
        self.assertRaises(ValueError, a.add, arr)

    def test_consistency_static_dynamic(self):
        """Test if container shape invariants hold."""
        a = Container()
        # add 10 rows static
        arr = np.empty((10, 0))
        a.add(arr)

        # add 11 rows dynamic
        arr = np.empty((11, 0))
        self.assertRaises(ValueError, a.add, arr, r=0, k=2)

        # add 9 rows dynamic
        arr = np.empty((9, 0))
        self.assertRaises(ValueError, a.add, arr, r=0, k=2)

        # add 11 rows static
        arr = np.empty((11, 0))
        self.assertRaises(ValueError, a.add, arr)

        # add 9 rows static
        arr = np.empty((9, 0))
        self.assertRaises(ValueError, a.add, arr)

    def test_consistency_combine_dense(self):
        """Test for consistency when combining containers with dense matrices. """
        self._check_consistency_combine(np.array)

    def test_consistency_combine_sparse(self):
        """Test for consistency when combining containers with sparse matrices. """
        self._check_consistency_combine(sp.csr_matrix)

    def _check_consistency_combine(self, array_factory):
        """Check for consistency when combining containers. """
        a = Container()
        arr = array_factory(np.empty((10, 1)))
        a.add(arr)

        # combine with dynamic only
        b = Container()
        arr = array_factory(np.empty((11, 1)))
        b.add(arr, r=0, k=2)
        self.assertRaises(ValueError, operator.add, a, b)

        # combine with static only
        b = Container()
        arr = array_factory(np.empty((11, 1)))
        b.add(arr)
        self.assertRaises(ValueError, operator.add, a, b)

    def test_save_and_load(self):
        """ Test saving and loading a Container object """
        c1 = Container()
        c1.add(np.array([[1,2,3],[4,5,6]]))
        c1.add(np.array([[2,3,4],[5,6,7]]), r=1, k=-1)
        c1.add(sp.csc_matrix([[3,4,5],[0,7,8]]))
        c1.add(sp.csc_matrix([[4,0,6],[7,8,9]]), r=2, k=-1)
        c1.save('testc')
        c2 = Container()
        c2.load('testc')
        self.assertTrue(np.all(c1.static_array==c2.static_array))
        self.assertTrue(np.all(c1.static_sparse.todense() == c2.static_sparse.todense()))
        self.assertEqual(c1.dynamic_array.keys(),c2.dynamic_array.keys())
        for k in c1.dynamic_array.keys():
            self.assertTrue(np.all(c1.dynamic_array[k] == c2.dynamic_array[k]))
        self.assertEqual(c1.dynamic_sparse.keys(),c2.dynamic_sparse.keys())
        for k in c1.dynamic_sparse.keys():
            self.assertTrue(np.all(c1.dynamic_sparse[k].todense() == c2.dynamic_sparse[k].todense()))
        self.assertEqual(c1.colnames(),c2.colnames())
        self.assertEqual(c1.colnames(r=1,k=-1),c2.colnames(r=1,k=-1))
        self.assertEqual(c1.colnames(r=2,k=-1),c2.colnames(r=2,k=-1))
        os.remove('testc.npz')
        os.remove('testc.mat')

    def test_metadata(self):
        c = Container()
        a_metadata = {1: 'B747', 3: 'A321', 4: 'A340'}
        c.add(np.array([[1,2,3],[4,5,6]]), colnames=list('abc'), metadata={'a': a_metadata})
        self.assertDictEqual(c.metadata()['a'], a_metadata)

        c2 = Container()
        d_metadata = {1: 'LOG', 2: 'VIE'}
        a_metadata = {1: 'B747', 3: 'A321', 4: 'A340'}
        # static
        c2.add(np.array([[1,2,3],[4,5,6]]), colnames=list('abc'), metadata={'a': a_metadata})
        # dynamic
        c2.add(np.array([[1,3,4],[2,6,7]]), r=1, k=-1, colnames=list('def'),
               metadata={'d': d_metadata})
        X = c2(r=1, k=-1)
        self.assertEqual(X.shape, (2, 6))
        colnames = c2.colnames(r=1, k=-1)
        self.assertEqual(colnames, list('abcdef'))

        # check the metadata
        metadata = c2.metadata(r=1, k=-1)
        self.assertEqual(metadata['a'], a_metadata)
        self.assertEqual(metadata['d'], d_metadata)

    def test_metadata_iadd(self):
        c1 = Container()
        a_metadata = {1: 'B747', 3: 'A321', 4: 'A340'}
        c1.add(np.array([[1,2,3],[4,5,6]]), colnames=list('abc'), metadata={'a': a_metadata})
        c = Container()
        c += c1
        self.assertDictEqual(c.metadata()['a'], a_metadata)

        c1 = Container()
        b_metadata = {1: 'LOG', 2: 'VIE'}
        c1.add(np.array([[1,2,3],[4,5,6]]), colnames=list('abc'), metadata={'a': a_metadata})
        c2 = Container()
        c2.add(np.array([[1,2,3],[4,5,6]]), colnames=list('abc'), metadata={'b': b_metadata})
        c2 += c1
        self.assertDictEqual(c2.metadata()['a'], a_metadata)
        self.assertDictEqual(c2.metadata()['b'], b_metadata)

        c1 = Container()
        b_metadata = {1: 'LOG', 2: 'VIE'}
        c1.add(np.array([[1,2,3],[4,5,6]]), colnames=list('abc'), metadata={'a': a_metadata})
        c2 = Container()
        c2.add(np.array([[1,2,3],[4,5,6]]), r=1, k=-1, colnames=list('abc'),
               metadata={'b': b_metadata})
        c2 += c1
        self.assertDictEqual(c2.metadata()['a'], a_metadata)
        self.assertRaises(KeyError, lambda col: c2.metadata()[col], 'b')
        self.assertDictEqual(c2.metadata(r=1, k=-1)['b'], b_metadata)

    def test_dynamic_metadata_merge(self):
        xcol1 = np.random.rand(10, 2)
        xcol2 = np.random.rand(10, 2)
        xcont1 = Container()
        xcont2 = Container()
        sevcols = {'importances-sev': [['sev-col1', 1], ['sev-col2', 2], ['sev-col3', 3]]}
        freqcols = {'coefficients-freq': [['freq-col1', 1], ['freq-col2', 2]]}
        xcont1.add(xcol1, metadata=sevcols, r=0, k=-1)
        xcont2.add(xcol2, metadata=freqcols,r=0, k=-1)
        xcont1.add(xcol1, metadata=sevcols, r=1, k=-1)
        xcont2.add(xcol2, metadata=freqcols,r=1, k=-1)
        self.assertEqual(xcont1.metadata(r=0, k=-1)['importances-sev'],
                         sevcols['importances-sev'])
        self.assertFalse('coefficients-freq' in xcont1.metadata(r=0, k=-1))

        xcont = xcont1 + xcont2
        self.assertEqual(xcont.metadata(r=0, k=-1)['importances-sev'],
                         sevcols['importances-sev'])
        self.assertTrue('coefficients-freq' in xcont.metadata(r=0, k=-1))

    def test_weights_and_offsets(self):
        weight = np.arange(10)
        cols = np.random.rand(10,2)
        offset = np.arange(10,20)

        #create a new container with some X data and a weight vector
        c1 = Container()
        c1.initialize({'weight':weight})
        c1.add(cols)

        #test initialization
        self.assertEqual(c1.modified, set())
        self.assertEqual(c1.variables.keys(), ['weight:None:None'])

        #test getting X data
        self.assertTrue(np.all(c1()==cols))

        #test getting the weight vector
        self.assertTrue(np.all(c1.get('weight')==weight))

        #verify that weight cannot be set/modified
        with self.assertRaises(ValueError):
            c1.set('weight', weight)

        #test the new function
        c2 = c1.new()
        #weight should be copied over but nothing else
        self.assertEqual(c1.variables, c2.variables)

        #verify value validation
        with self.assertRaises(ValueError):
            c2.set('offset', {})

        #add offset
        c2.set('offset', offset)
        self.assertTrue(np.all(c2.get('offset')==offset))
        self.assertEqual(c2.modified, {'offset'})

        #combine containers
        c3 = c1+c2

        self.assertTrue(np.all(c3()==cols))
        self.assertTrue(np.all(c3.get('weight')==weight))
        self.assertTrue(np.all(c3.get('offset')==offset))
        self.assertEqual(c3.modified, {'offset'})

        #add to container
        self.assertEqual(c1.modified, set())
        self.assertEqual(c1.get('offset'), None)

        c1+=c2

        self.assertTrue(np.all(c1()==cols))
        self.assertTrue(np.all(c1.get('weight')==weight))
        self.assertTrue(np.all(c1.get('offset')==offset))
        self.assertEqual(c1.modified, {'offset'})

        #check remaining container
        self.assertEqual(c2().shape, (0,0))
        self.assertTrue(np.all(c2.get('offset')==offset))
        self.assertEqual(c2.modified, {'offset'})

        #set arbitrary key
        c2.set('blah', offset)
        self.assertTrue(np.all(c2.get('blah')==offset))
        self.assertEqual(c3.get('blah'),None)

        #check variable keys
        self.assertEqual(c1.variable_keys(), {'weight','offset'})
        self.assertEqual(c2.variable_keys(), {'weight','offset','blah'})

    def test_check_output_weights(self):
        weight = np.arange(10)
        cols = np.random.rand(10,2)
        cols2 = np.random.rand(10,2)
        offset = np.arange(10,20)

        #create a new container with some X data and a weight vector
        c1 = Container()
        c1.initialize({'weight':weight})
        c1.add(cols)

        c2 = Container()
        c2.add(cols2)

        # action
        out = check_output_weights(c1, c2)
        self.assertTrue(np.all(out()==cols2))
        self.assertEqual(out.variables, c1.variables)
        self.assertEqual(out.modified, c1.modified)

    def test_balance_weights(self):
        weight = pd.Series(np.arange(10))
        Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        cols = np.random.rand(10,2)

        #create a new container with some X data and a weight vector
        c1 = Container()
        c1.initialize({'weight':weight})
        c1.add(cols)

        c1.balance('weight', Y)
        #check balanced weights exist and are balanced
        self.assertIn('balanced_weight:None:None', c1.variables)
        values = c1.variables['balanced_weight:None:None']
        self.assertEqual(np.sum(values[:5]), np.sum(values[5:]))

        #make sure balanced weights are not re-calculated
        values = c1.variables['balanced_weight:None:None']
        values[0] = 999
        c1.balance('weight', Y)
        self.assertNotEqual(np.sum(values[:5]), np.sum(values[5:]))


class ImmutableDictViewTest(unittest.TestCase):

    def test_view_iter(self):
        v = _ImmutableDictView({'a': 0}, {'b': 1})
        keys = [k for k in v]
        self.assertEqual(set(keys), set(v.keys()))
        self.assertEqual(set(keys), set(v.iterkeys()))
        self.assertEqual(v.values(), list(v.itervalues()))


class SafeHStackTest(unittest.TestCase):

    def test_smoke(self):
        col = np.arange(10)[:, np.newaxis]
        A = safe_hstack((col, col, col))
        self.assertEqual(A.shape, (col.shape[0], 3))
        np.testing.assert_array_equal(A, col.repeat(3, axis=1))

    def test_empty(self):
        col = np.empty((10, 0))
        A = safe_hstack((col, col, col))
        self.assertEqual(A.shape, (col.shape[0], 0))

    def test_smoke_sparse(self):
        col = np.arange(10)[:, np.newaxis]
        A = safe_hstack((col, col, sp.csr_matrix(col)))
        self.assertEqual(A.shape, (col.shape[0], 3))
        self.assertTrue(sp.isspmatrix(A))
        np.testing.assert_array_equal(A.toarray(), col.repeat(3, axis=1))

    def test_empty_sparse(self):
        col = np.empty((10, 0))
        A = safe_hstack((col, col, sp.csr_matrix(col)))
        self.assertEqual(A.shape, (col.shape[0], 0))
        self.assertTrue(sp.isspmatrix(A))

class TestDeepUpdate(unittest.TestCase):
    def test_deep_update1(self):
        a = {'a':{'b':1, 'c':2}, 'd':{'e':3}, 'f':[1,2,3]}
        b = {'a':{'f':4}, 'g':5}
        expected = {'a':{'b':1,'c':2,'f':4}, 'd':{'e':3}, 'f':[1,2,3], 'g':5}
        _deep_update(a,b)
        self.assertEqual(a, expected)

        a = {'a':{'b':{'c':{'d':1, 'e':2}}}}
        b = {'a':{'b':{'c':{'e':3, 'f':4}}}}
        expected = {'a':{'b':{'c':{'d':1, 'e':3, 'f':4}}}}
        _deep_update(a,b)
        self.assertEqual(a, expected)



if __name__=='__main__':
    unittest.main()
