#########################################################
#
#       Unit Test for Utilities
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import random
import unittest
import pandas as pd
import numpy as np
import pytest

import ModelingMachine.engine.utilities as blueprint_module
from ModelingMachine.engine.utilities import string_pop, validate_blueprint, BlueprintHelper
from ModelingMachine.engine.data_utils import array_hash, toplevels
from ModelingMachine.engine.pandas_data_utils import varTypeString
from ModelingMachine.engine.plan import menu as menu_bin
from ModelingMachine.engine.plan_reg import menu_reg

import ModelingMachine.engine.pandas_data_utils as pdu

class TestUtilities(unittest.TestCase):
    def test_string_pop(self):
        x = 'asdfghjk'
        self.assertEqual(string_pop(x,0),'sdfghjk')
        self.assertEqual(string_pop(x,1),'adfghjk')
        self.assertEqual(string_pop(x,3),'asdghjk')
        self.assertEqual(string_pop(x,7),'asdfghj')
        self.assertEqual(string_pop(x,-1),'asdfghj')
        self.assertEqual(string_pop(x,-2),'asdfghk')
        self.assertEqual(string_pop(x,-7),'adfghjk')
        self.assertEqual(string_pop(x,-8),'sdfghjk')
        self.assertRaises(Exception,string_pop,x,8)
        self.assertRaises(Exception,string_pop,x,-9)

    def test_arrayhash(self):
        x = np.array([[1,2,3],[4,5,6],[7,8,9]])
        y = np.array([range(5),range(5)])
        a = array_hash(x)
        b = array_hash(x)
        c = array_hash(y)
        self.assertIsInstance(a,str)
        self.assertEqual(len(a),40)
        self.assertEqual(a,b)
        self.assertNotEqual(b,c)

    def test_toplevels(self):
        x = np.concatenate((np.repeat(['a'], 40), np.array([np.NaN, 'c', 'b']), np.repeat(['d'], 21), np.array([np.NaN, 'c', 'f', 'b', np.NaN]), np.repeat(['g'], 21), np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])))
        self.assertEqual(len(x), 100)
        xs = pd.Series(x)
        xs[xs == 'nan'] = np.NaN
        tl1 = toplevels(xs, n=1)
        self.assertEqual(sorted(tl1.unique().tolist()), ['==Missing==', '=All Other='])
        tl2 = toplevels(xs, n=3)
        self.assertEqual(sorted(tl2.unique().tolist()), ['==Missing==', '=All Other=', 'a', 'g']) 
        tl3 = toplevels(xs, n=5)
        self.assertEqual(sorted(tl3.unique().tolist()), ['==Missing==', '=All Other=', 'a', 'b', 'd', 'g'])
        tl4 = toplevels(xs, pct_thresh=0.40)
        self.assertEqual(sorted(tl4.unique().tolist()), ['==Missing==', '=All Other=', 'a'])
        tl5 = toplevels(xs, pct_thresh=0.82)
        self.assertEqual(sorted(tl5.unique().tolist()), ['==Missing==', '=All Other=', 'a', 'd', 'g'])
        tl6 = toplevels(xs, pct_thresh=0.82, n=2)
        self.assertEqual(sorted(tl6.unique().tolist()), ['==Missing==', '=All Other=', 'a'])

    def test_invalid_bps(self):
        #not dict
        bp = []
        self.assertRaises(Exception,validate_blueprint,bp)
        #zero length
        bp = {}
        self.assertRaises(Exception,validate_blueprint,bp)
        #too long (length capped at 99)
        bp = dict(zip(range(999),range(999)))
        self.assertRaises(Exception,validate_blueprint,bp)
        #vertex doesn't have three elements
        bp = {'1':(['NUM'],'T'),'2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #vertex task list is empty
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],[],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #first vertex input is not a list
        bp = {'1':('NUM',['NI'],'T'),'2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #first vertex task is not a converter
        bp = {'1':(['NUM'],['ST'],'T'),'2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #first vertex task is an invalid converter
        bp = {'1':(['NUM'],['TM2'],'T'),'2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #first vertex input is invalid
        bp = {'1':(['NUM','1'],['NI'],'T'),'2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #first vertex task is an invalid converter
        bp = {'1':(['NUM','CAT'],['NI'],'T'),'2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #no primary vertices
        bp = {'1':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #invalid input in second vertex
        bp = {'1':(['NUM'],['NI'],'T'),'2':(['2'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #invalid method:
        bp = {'1':(['NUM'],['NI'],'F'), '2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #final method not equal to P
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],['RFI'],'T')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #dead end: second vertex is not utilized
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['CAT'],['DM2'],'T'), '3':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #invalid vertex tasks
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],['INVALIDTASK'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #second vertex task list is not a list
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],'RFI','P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #invalid blueprint key
        bp = {'1':(['NUM'],['NI'],'T'), '3':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #invalid input for converter
        bp = {'1':(['CAT'],['NI'],'T'), '2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #two converters
        bp = {'1':(['NUM'],['NI','NI'],'T'), '2':(['1'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)
        #converter in non primary vertex
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],['NI','RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,bp)

    def test_valid_bps(self):
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],['RFI'],'P')}
        self.assertTrue(validate_blueprint(bp))
        bp = {'1':(['NUM'],['LS','ST'],'T'), '2':(['1'],['LR1'],'P')}
        self.assertTrue(validate_blueprint(bp))
        bp = {'1':(['NUM'],['LS'],'T'), '2':(['1'],['ST','LR1'],'P')}
        self.assertTrue(validate_blueprint(bp))
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['CAT'],['DM2'],'T'), '3':(['1','2'],['RFI'],'P')}
        self.assertTrue(validate_blueprint(bp))
        bp = {'1':(['CAT'],['DM2'],'T'), '2':(['1'],['ST','LR1'],'T'), '3':(['NUM'],['NI'],'T') ,'4':(['2','3'],['RFI'],'P')}
        self.assertTrue(validate_blueprint(bp))
        bp = {'1':(['CAT'],['DM2'],'T'), '2':(['TXT'],['TM2','LR1'],'T'), '3':(['NUM'],['NI'],'T') ,'4':(['1','2','3'],['RFI'],'P')}
        self.assertTrue(validate_blueprint(bp))
        menu = menu_bin+menu_reg
        for item in menu:
            bp = item['blueprint']
            self.assertTrue(validate_blueprint(bp))

    def test_with_available_types(self):
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],['RFI'],'P')}
        self.assertTrue(validate_blueprint(bp,['N']))
        self.assertTrue(validate_blueprint(bp,['N','C']))
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['CAT'],['DM2'],'T'), '3':(['1','2'],['RFI'],'P')}
        self.assertRaises(Exception,validate_blueprint,[bp,['N']])
        bp = {'1':(['CAT'],['DM2'],'T'), '2':(['TXT'],['TM2','LR1'],'T'), '3':(['NUM'],['NI'],'T') ,'4':(['1','2','3'],['RFI'],'P')}
        self.assertTrue(validate_blueprint(bp,['N','C','T']))

    def test_task_parameters(self):
        bp = {'1':(['NUM'],['NI'],'T'), '2':(['1'],['RFC nt=123;ls=12'],'P')}
        blueprint = BlueprintHelper(bp)
        out = blueprint.task_parameters()
        self.assertEqual(out['RFC nt=123;ls=12']['n_estimators'],'123')
        self.assertEqual(out['RFC nt=123;ls=12']['min_samples_leaf'],'12')

    def test_BIN_V1_is_valid(self):
        bp = {'1':(['BIN_V1'],['PASS'],'T'),
              '2':(['1'],['RFR'], 'P')}
        validate_blueprint(bp)
        #Passes if validation does not raise AssertionError


class TestPandasUtilities(unittest.TestCase):

    def test_getX_with_CAT_V1(self):
        d = pd.DataFrame({'a':np.linspace(0,1,10),
                          'b':[str(i) for i in xrange(10)],
                          'c':np.random.randn(10) })
        varTypes = {'NUM':[2], 'CAT_V1':[0,1]}
        vtype = 'CAT_V1'

        X = pdu.getX(d, 'c', varTypes, {}, vtype=vtype)

        for dt in X.dtypes:
            self.assertEqual(dt, 'O') #Assert all objects i.e. strings

    def test_getX_dates(self):
        inp = pd.DataFrame({'a': ['1:30', '12:12', '14:00'],
                            'b': [str(i) for i in xrange(3)],
                            'c': ['2007-3-1 1:30:12', '2007-4-20 12:12:00', '2009-12-2 14:14:01'],
                            'd': ['12/4/1996', '4/5/2000', '5/6/2007'] })
        out = pd.DataFrame({'a': [ -2208965400, -2208926880, -2208920400 ],
                            'c': [ 1172730612, 1177085520, 1259781241 ],
                            'd': [ 728997, 730215, 732802] })

        vt,tc = varTypeString(inp)
        X = pdu.getX(inp, 'b', None, tc)
        self.assertTrue(all(X==out))
        self.assertTrue(all(X.dtypes==['float64','float64','int64']))

    def test_getX_percent(self):
        inp = pd.DataFrame({'a': ['100%', '31.1 %', '-4%'],
                            'b': [str(i) for i in xrange(3)] })

        X = pdu.getX(inp, 'b', None, {'a':'P'})
        self.assertTrue(all(X==[100, 31.1, -4]))
        self.assertTrue(all(X.dtypes==['float64',]))

    def test_getX_currency(self):
        inp = pd.DataFrame({'a': ['$100.99', '50 $', '-4 EUR'],
                            'b': [str(i) for i in xrange(3)] })

        X = pdu.getX(inp, 'b', None, {'a':'$'})
        self.assertTrue(all(X==[100.99, 50, -4]))
        self.assertTrue(all(X.dtypes==['float64',]))

    def test_getX_currency_ranges_are_string(self):
        """Test that app loads with string data"""
        inp = pd.DataFrame({'a': [ "$500;000-$774;999", "$500;000-$774;999",
                "$300;000-$349;999", "$300;000-$349;999",  "$300;000-$349;999",
                "$300;000-$349;999",  "$450;000-$499;999", "$250;000-$274;999",
                "$175;000-$199;999",  "$175;000-$199;999"],
            'b': [str(i) for i in xrange(10)] })
        X = pdu.getX(inp, 'b', None, {'a':'X'})
        print X.dtypes
        self.assertTrue(all(X.dtypes==['object',]))

@pytest.mark.unit
class TestSubBlueprints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bp = {'1': (['NUM'], ['LS', 'CCZ'], 'T'),
                  '2': (['CAT'], ['DM2', 'CCZ'], 'T'),
                  '3': (['TXT'], ['TM2', 'LR1'], 'S'),
                  '4': (['2'], ['ST', 'LR1'], 'T'),
                  '5': (['1', '3', '4'], ['ST', 'PCA', 'SVCR'], 'P')}

    def test_sub_does_not_return_self(self):
        subs = blueprint_module.sub_blueprints_of(self.bp)

        for sub in subs:
            self.assertNotIn('5', sub)

    def test_sub_returns_fixed_vertices(self):
        subs = blueprint_module.sub_blueprints_of(self.bp)

        for sub in subs:
            keys = sub.keys()
            self.assertEqual(sorted(keys), [str(i) for i in range(1, len(keys)+1)])

    def test_fix_connectivity_does_not_change_valid(self):
        fixed = blueprint_module.fix_connectivity(self.bp)

        self.assertEqual(sorted(fixed.keys()), sorted(self.bp.keys()))
        for key in fixed.keys():
            self.assertEqual(fixed[key], self.bp[key])

    def test_fix_connectivity(self):
        broken = {'2':( ['NUM'], ['NI', 'CCZ'], 'T' ),
                  '4':( ['2'], ['LR1'], 'P' ) }

        fixed = blueprint_module.fix_connectivity(broken)
        self.assertEqual(sorted(fixed.keys()), [str(i) for i in range(1, len(broken)+1)])

    def test_valid_sub_blueprint(self):
        out = blueprint_module.valid_sub_blueprints(self.bp)
        for i in out:
            self.assertTrue(validate_blueprint(i))
        answer = [{'1': (['TXT'], ['TM2', 'LR1'], 'P')},
                {'1': (['CAT'], ['DM2', 'CCZ'], 'T'), '2': (['1'], ['ST', 'LR1'], 'P')}]
        self.assertEqual(out,answer)





if __name__ == '__main__':
    unittest.main()
