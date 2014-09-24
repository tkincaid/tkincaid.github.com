import unittest
import random
import pandas
import numpy as np

from ModelingMachine.engine.tasks.cfconverter import make_level_map
from ModelingMachine.engine.tasks.cfconverter import map_levels
from ModelingMachine.engine.tasks.cfconverter import CFConverter
from ModelingMachine.engine.tasks.cfconverter import CFCATConverter
from ModelingMachine.engine.tasks.cfconverter import CFCATCred
from ModelingMachine.engine.tasks.cfconverter import CFCATCount
from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.response import Response
from ModelingMachine.engine.container import Container

class BaseCFConverterTest(unittest.TestCase):

    def create_string_array(self, n_labels, size, seed=1):
        labels = map(lambda x:'A{}'.format(x), range(n_labels))
        random.seed(seed)
        return pandas.Series([random.choice(labels) for i in range(size)])

    def create_numeric_array(self, n_labels, size, seed=1):
        labels = range(n_labels)
        random.seed(seed)
        return pandas.Series([float(random.choice(labels)) for i in range(size)])

    def create_dataframe(self, size, varTypeString, seed=1, categorical_cf=True, n_labels=10):
        out = {}
        for n,i in enumerate(list(varTypeString)):
            if i=='N':
                out['var{}'.format(n)] = self.create_numeric_array(n_labels, size, seed=seed * n)
            elif i in ('U', 'I'):
                if categorical_cf:
                    out['var{}'.format(n)] = self.create_string_array(n_labels, size, seed=seed * n)
                else:
                    out['var{}'.format(n)] = self.create_numeric_array(n_labels, size, seed=seed * n)
            else:
                out['var{}'.format(n)] = self.create_string_array(n_labels, size, seed=seed * n)
        ds = pandas.DataFrame(out)
        ds.special_columns = {}
        for n,i in enumerate(list(varTypeString)):
            if i in ['U','I']:
                ds.special_columns[i] = ds.columns[n]
        return ds

    def create_YZ(self, size, seed=1, ):
        Y = np.random.randn(size, 1)
        Z = Partition(size=size, folds=1, reps=1, total_size=size)
        Z.set(max_reps=1, max_folds=0)
        return Y, Z


class TestCFConverter(BaseCFConverterTest):

    def test_make_level_map(self):
        x = self.create_string_array(20, 10000)

        out = make_level_map(x)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), set(x))
        self.assertEqual(sorted(out.values()), range(len(out)))

        out = make_level_map(x, 40)
        self.assertIsInstance(out, dict)
        self.assertEqual(set(out), set(x))
        self.assertEqual(sorted(out.values()), range(40, 40+len(out)) )

    def test_map_levels(self):
        x = self.create_string_array(10, 100)

        level_map = make_level_map(x)

        out, outmap = map_levels(x, level_map)

        self.assertEqual(outmap, level_map)
        for n,val in enumerate(x):
            self.assertEqual(out[n], outmap[val])

        #include some new levels
        x = self.create_string_array(20, 100)

        out, outmap = map_levels(x, level_map)

        self.assertEqual(outmap, level_map)
        # new labels are mapped to -1
        for o, i in zip(out, x):
            assert (True if i in level_map else o == -1)

    def test_fit_transform_smoke(self):
        x = self.create_dataframe(100, 'NNCUI')
        cf = CFConverter()
        Y = Response.from_array(np.random.rand(100))
        Z = Partition(100, seed=1)
        Z.set(partitions=[(0, -1)])
        cf.fit(Container(x), Y, Z)
        out = cf.transform(Container(x), Y, Z)
        for p in Z:
            ctx = out.get_user_item_context(**p)
            self.assertEqual(ctx.user_id, 0)
            self.assertEqual(ctx.item_id, 1)
            self.assertEqual(ctx.n_users, 10)
            self.assertEqual(ctx.n_items, 10)

    def test_fit_transform_only_item(self):
        x = self.create_dataframe(100, 'NCIT')
        cf = CFConverter()
        Y = Response.from_array(np.random.rand(100))
        Z = Partition(100, seed=1)
        Z.set(partitions=[(0, -1)])
        cf.fit(Container(x), Y, Z)
        out = cf.transform(Container(x), Y, Z)
        for p in Z:
            ctx = out.get_user_item_context(**p)
            self.assertEqual(ctx.user_id, None)
            self.assertEqual(ctx.item_id, 0)
            self.assertEqual(ctx.n_users, None)
            self.assertEqual(ctx.n_items, 10)

    def test_no_cf_inputs(self):
        x = self.create_dataframe(100, 'NNCT')
        cf = CFConverter()
        Y = Response.from_array(np.random.rand(100))
        Z = Partition(100, seed=1)
        cf.fit(Container(x), Y, Z)
        out = cf.transform(Container(x), Y, Z)
        for p in Z:
            key = (p['r'], p['k'])
            self.assertEqual(cf.data[key], {})

            ctx = out.get_user_item_context(**p)
            self.assertEqual(ctx.user_id, None)
            self.assertEqual(ctx.item_id, None)
            self.assertEqual(ctx.n_users, None)
            self.assertEqual(ctx.n_items, None)

    def test_not_fitted(self):
        x = self.create_dataframe(100, 'UNNCT')
        Y = Response.from_array(np.random.rand(100))
        Z = Partition(100, seed=1)
        cf = CFConverter()
        self.assertRaises(ValueError, cf.transform, Container(x), Y, Z)


class TestCFCATConverter(BaseCFConverterTest):
    """Test cases for our CAT-CF One-hot encoder. """

    def test_fit_smoke(self):
        x = self.create_dataframe(100, 'CU')
        Y, Z = self.create_YZ(100)
        cf = CFCATConverter('sc=0;cm=10000')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x))
        self.assertEqual(o().shape, (100, 61))
        self.assertEqual(o.colnames(),
                         [u'var0-A0__A0', u'var0-A0__A5', u'var0-A0__A8', u'var0-A0__A9', u'var0-A1__A0', u'var0-A1__A4', u'var0-A1__A9', u'var0-A2__A2', u'var0-A2__A3', u'var0-A2__A7', u'var0-A2__A8', u'var0-A2__A9', u'var0-A3__A0', u'var0-A3__A2', u'var0-A3__A5', u'var0-A3__A7', u'var0-A3__A8', u'var0-A4__A0', u'var0-A4__A2', u'var0-A4__A4', u'var0-A4__A5', u'var0-A4__A7', u'var0-A4__A8', u'var0-A4__A9', u'var0-A5__A0', u'var0-A5__A2', u'var0-A5__A3', u'var0-A5__A4', u'var0-A5__A5', u'var0-A5__A8', u'var0-A6__A0', u'var0-A6__A1', u'var0-A6__A2', u'var0-A6__A3', u'var0-A6__A4', u'var0-A6__A5', u'var0-A6__A6', u'var0-A6__A7', u'var0-A6__A8', u'var0-A7__A0', u'var0-A7__A3', u'var0-A7__A5', u'var0-A7__A6', u'var0-A7__A7', u'var0-A7__A8', u'var0-A8__A1', u'var0-A8__A2', u'var0-A8__A3', u'var0-A8__A4', u'var0-A8__A5', u'var0-A8__A6', u'var0-A8__A7', u'var0-A8__A9', u'var0-A9__A0', u'var0-A9__A2', u'var0-A9__A4', u'var0-A9__A5', u'var0-A9__A6', u'var0-A9__A7', u'var0-A9__A8', u'var0-A9__A9'])
        # case where doesn't contain any additional info
        x.iloc[:,0] = x.iloc[:,1]
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (0, 0))

    def test_fit_multiple_cat(self):
        x = self.create_dataframe(100, 'CCIU')
        Y, Z = self.create_YZ(100)
        cf = CFCATConverter('sc=0;cm=10000')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x))
        self.assertEqual(o().shape, (100, 247))

    def test_fit_num(self):
        x = self.create_dataframe(100, 'NIU')
        Y, Z = self.create_YZ(100)
        cf = CFCATConverter('sc=0;cm=10000')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x))
        self.assertEqual(o().shape, (100, 120))

    def test_fit_num_cf(self):
        x = self.create_dataframe(100, 'CIU', categorical_cf=False)
        Y, Z = self.create_YZ(100)
        cf = CFCATConverter('sc=0;cm=10000')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x))
        self.assertEqual(o().shape, (100, 120))


class TestCFCATCred(BaseCFConverterTest):
    """Test cases for our CAT-CF credibility estimates. """

    def test_fit_smoke(self):
        x = self.create_dataframe(100, 'CU')
        Y, Z = self.create_YZ(100)
        cf = CFCATCred('sc=0;cmin=1')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 1))
            self.assertEqual(o.colnames(**p), ['DR_cred_user_var0'])
        o = cf.transformer_stack(Container(x), Y, Z)
        for p in Z:
            print o(**p)
            self.assertEqual(o(**p).shape, (100, 1))
            self.assertEqual(o.colnames(**p), ['DR_cred_user_var0'])
        # case where doesn't contain any additional info
        x.iloc[:,0] = x.iloc[:,1]
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (0, 0))

    def test_fit_multiple_cat(self):
        x = self.create_dataframe(100, 'CCIU')
        Y, Z = self.create_YZ(100)
        cf = CFCATCred('sc=0;cmin=1')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 4))
            self.assertEqual(o.colnames(**p), ['DR_cred_item_var0', 'DR_cred_item_var1', 'DR_cred_user_var0', 'DR_cred_user_var1'])

    def test_fit_num(self):
        x = self.create_dataframe(100, 'NIU')
        Y, Z = self.create_YZ(100)
        cf = CFCATCred('sc=0;cmin=1')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 2))
            self.assertEqual(o.colnames(**p), ['DR_cred_item_var0', 'DR_cred_user_var0'])

    def test_fit_num_cf(self):
        x = self.create_dataframe(100, 'CIU', categorical_cf=False)
        Y, Z = self.create_YZ(100)
        cf = CFCATCred('sc=0;cmin=1')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 2))
            self.assertEqual(o.colnames(**p), ['DR_cred_item_var0', 'DR_cred_user_var0'])

class TestCFCATCount(BaseCFConverterTest):
    """Test cases for our CAT-CF credibility estimates. """

    def test_fit_smoke(self):
        x = self.create_dataframe(100, 'CU')
        Y, Z = self.create_YZ(100)
        cf = CFCATCount('cmin=100')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 1))
            self.assertEqual(o.colnames(**p), ['user_var0_count'])
        # case where doesn't contain any additional info
        x.iloc[:,0] = x.iloc[:,1]
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (0, 0))

    def test_fit_multiple_cat(self):
        x = self.create_dataframe(100, 'CCIU')
        Y, Z = self.create_YZ(100)
        cf = CFCATCount('cmin=1')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 4))
            self.assertEqual(o.colnames(**p), ['item_var0_count', 'item_var1_count', 'user_var0_count', 'user_var1_count'])

    def test_fit_num(self):
        x = self.create_dataframe(100, 'NIU')
        Y, Z = self.create_YZ(100)
        cf = CFCATCount('cmin=1')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 2))
            self.assertEqual(o.colnames(**p), ['item_var0_count', 'user_var0_count'])

    def test_fit_num_cf(self):
        x = self.create_dataframe(100, 'CIU', categorical_cf=False)
        Y, Z = self.create_YZ(100)
        cf = CFCATCount('cmin=1')
        cf.fit(Container(x), Y, Z)
        o = cf.transform(Container(x), Y, Z)
        for p in Z:
            self.assertEqual(o(**p).shape, (100, 2))
            self.assertEqual(o.colnames(**p), ['item_var0_count', 'user_var0_count'])

if __name__=='__main__':
    unittest.main()


