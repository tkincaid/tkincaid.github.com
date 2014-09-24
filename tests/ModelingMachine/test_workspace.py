############################################################################
#
#       unit test for Workspace
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import pandas
import numpy as np
import unittest
import hashlib
import copy
from bson.objectid import ObjectId
import os
import time
from mock import patch
import numbers
import random
import pandas as pd

from config.engine import EngConfig
import config.test_config

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.response import Response
from ModelingMachine.engine.pandas_data_utils import getY, varTypes, varTypeString
from ModelingMachine.engine.workspace import Workspace
from ModelingMachine.engine.vertex import Vertex
import common.storage as storage

from MMApp.entities.db_conn import DBConnections
from tests.IntegrationTests.storage_test_base import StorageTestBase
from common.services.flippers import GlobalFlipper
FLIPPERS = GlobalFlipper()

class TestWorkspace(StorageTestBase):
    def np_equal(self,a,b):
        """ assert two numpy arrays are equal (even if they have nan's)
        """
        try:
            np.testing.assert_equal(a,b)
        except AssertionError:
            return False
        return True

    @classmethod
    def setUpClass(cls):
        super(TestWorkspace, cls).setUpClass()
        # Call to method in StorageTestBase
        cls.datasets = [
            'allstate-nonzero-200.csv',
            'kickcars.rawdata.csv',
            'credit-sample-200.csv',
            'fastiron-train-sample-small.csv',
            'kickcars-sample-200.csv',
            'credit-na-test-200.csv',
            'low_info.csv',
            'dup_test.csv'
        ]

        cls.test_directory, _ = cls.create_test_files(cls.datasets)

        cls.datasets = cls.datasets[:3]

        cls.testdatafile = 'credit-sample-200.csv' # The default
        cls.targetnames = ['Claim_Amount','y','SeriousDlqin2yrs']
        cls.newdata = {'filename':cls.testdatafile,'target':{'name':'SeriousDlqin2yrs','type':'Binary','size':200}}

    def setUp(self):
        super(TestWorkspace, self).setUp()
        #setup a test project
        self.db_conn = DBConnections()
        self.collection = self.db_conn.get_collection("project")
        self.workspace = Workspace()
        self.workspace.db = self.db_conn.get_collection
        self.workspace.location = self.test_directory
        self.workspace.init_project({'filename': 'projects/'+str(self.pid)+'/raw/'+self.testdatafile,'uid':ObjectId(), 'pid': self.pid, 'active':1,'stage':'eda:',
            'originalName':self.testdatafile,'created':time.time(),'holdout_pct':0})
        self.workspace.init_dataset("universe", ['projects/'+str(self.pid)+'/raw/'+self.testdatafile])

    def tearDown(self):
        super(TestWorkspace, self).tearDown()
        #remove the project
        self.db_conn.get_collection('project').remove()
        self.db_conn.get_collection('metadata').remove()


    def init_workspace(self, datasetname):
        '''Initialize self.pid, and self.workspace to operate on the provided
        dataset.  You should make sure that datasetname is inside the
        enumeration in setUpClass so that it gets copied into the temporary
        directory
        '''
        self.workspace = Workspace()
        self.workspace.db = self.db_conn.get_collection
        self.workspace.location = self.test_directory
        self.workspace.init_project({'filename':'projects/'+str(self.pid)+'/raw/'+datasetname,'uid':ObjectId(), 'pid':self.pid,'active':1,'stage':'eda:',
            'originalName':datasetname,'created':time.time(),'holdout_pct':0})
        self.workspace.init_dataset("universe", ['projects/'+str(self.pid)+'/raw/'+datasetname])

    def test_set_target(self):
        self.assertRaises(AssertionError,self.workspace.set_target,0)
        ds = self.workspace.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        testtarget = {'name':targetname,'type':'Binary','size':200}
        self.workspace.set_target(targetname)
        query_cursor = self.collection.find({'_id':self.pid})
        query = [i for i in query_cursor]
        self.assertEqual(len(query),1)
        query = query[0]
        target = query['target']
        self.assertEqual(target,testtarget)
        #test with verify=True - in a separate test below

    def test_get_target(self):
        print self.workspace.data
        ds = self.workspace.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        target = {'name':targetname,'type':'Binary','size':200}
        self.workspace.set_target(targetname)
        self.assertEqual(self.workspace.get_target(),target)
        self.assertEqual(self.workspace.data.get('target'),target) #check cache populated
        self.collection.remove({'_id':self.pid}) #empty database
        self.workspace.data = {} #empty cache
        self.assertEqual(self.workspace.get_target(),None)
        self.assertEqual(self.workspace.data,{}) #check that workspace.data was not altered

    def test_names(self):
        self.assertEqual(self.workspace._filepath('test'), os.path.join(self.workspace.location,'test'))
        #self.assertEqual(self.workspace._get_filename(),self.testdatafile)
        self.assertEqual(self.workspace._get_filename(), self.workspace.dataset_id)
        namemd5 = hashlib.md5(self.workspace.dataset_id).hexdigest()
        self.assertEqual(self.workspace._get_dataset_name(),str(self.pid)+'-'+namemd5)
        self.assertEqual(self.workspace._get_vertex_name('asdf',None,{'r':0,'k':-1}),str(self.pid)+'-'+namemd5+'-asdf-1-0')

    def test_csv_to_dataframe(self):
        ds = self.workspace.get_dataframe()
        self.assertIsInstance(ds,pandas.DataFrame)
        self.assertEqual(ds.shape,(200,12))
        filename = self.workspace._get_dataset_name()
        fo = storage.FileObject(filename)
        self.assertFalse(fo.exists()) #check file doesnt exist

    def test_get_dataframe_wrongfile(self):
        self.assertRaises(ValueError,self.workspace.get_dataframe,'wrongfilename')

    def test_new_init(self):
        class WorkspaceTest(Workspace):
            db = self.db_conn.get_collection
            location = self.test_directory
        workspace = WorkspaceTest()
        pid = ObjectId()
        workspace.init_project({'filename': 'projects/'+str(self.pid)+'/raw/'+self.newdata['filename'],'target':self.newdata['target'], 'pid':pid})
        target = self.newdata['target']
        workspace.data = {}
        self.assertEqual(workspace.get_target(),target)
        query_cursor = self.collection.find({'_id':workspace.pid})
        query = [i for i in query_cursor]
        self.assertEqual(len(query),1)
        self.assertEqual(query[0].get('target'),target)

    def test_varTypeString(self):
        answers = ['NNNNNCCCCCCCCCCCCCCCNNNNNNNNNCNNNNN','NNCNNCCCCCCNCNCCCNNNNNNNNCCNNCNNNNC','NNNNNNNNNNNN']
        for i,f in enumerate(self.datasets):
            local_file = os.path.join(self.test_directory,f)
            print f
            storage.FileObject(f).get(local_file)
            ds = pandas.read_csv(local_file)
            vtypes,tc = varTypeString(ds)
            self.assertEqual(vtypes,answers[i])
            #the next lines allow human inspection of the answers
            #for n,col in enumerate(ds):
            #    print col+' == '+vtypes[n]

    def test_target_verify_and_metadata(self):
        ds = self.workspace.get_dataframe()
        #put metadata into DB
        #self.workspace.put_dataframe(ds)
        #test metadata by querying from DB
        self.workspace._query_db()
        self.assertNotIn('target',set(self.workspace.data.keys())) #target not set yet
        self.workspace._query_metadata()
        metadict = self.workspace.datasets
        filename = self.workspace._get_filename()
        self.assertIn(filename, metadict.keys())
        meta = metadict[filename]
        self.assertEqual(meta.get('shape'),list(ds.shape))
        #self.assertEqual(meta.get('columns'),list(ds.columns))
        # vartypes now in EDA
        #self.assertEqual(meta.get('varTypeString'),varTypeString(ds)[0])
        self.assertEqual(meta.get('pid'),self.workspace.pid)
        #test set target verify=True
        self.assertRaises(AssertionError,self.workspace.set_target,'wrongname')
        targetname = 'SeriousDlqin2yrs'
        target = {'name':targetname,'type':'Binary','size':200}
        self.workspace. set_target(targetname)
        check = self.workspace.get_target()
        self.assertEqual(target,check)

    def test_get_metadata(self):
        #workspace has no metadata yet
        #self.assertRaises(ValueError,self.workspace.get_metadata)
        #create metadata
        ds = self.workspace.get_dataframe()
        #self.workspace.put_dataframe(ds)
        #wrong filename
        self.assertRaises(ValueError,self.workspace.get_metadata,'wrongname')
        #valid query
        meta = self.workspace.get_metadata()
        self.assertIsInstance(meta,dict)
        self.assertEqual(set(meta.keys()),set(['_id', 'varTypeString','typeConvert','columns','shape','pid','created',
                                               'files','originalName','name','controls']))
        # var types are now part of EDA
        #self.assertEqual(meta['varTypeString'],varTypeString(ds)[0])
        self.assertEqual(meta['shape'],list(ds.shape))
        #self.assertEqual(meta['columns'],list(ds.columns))

    def test_apply_target_options_float_binary_with_no_missing_data(self):
        '''Use credit-200 with target SeriousDlqin2yrs - already encoded
        as 0/1, no missing variables

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        self.init_workspace('credit-sample-200.csv')
        ds = self.workspace.get_dataframe().copy()
        target_data = self.workspace.data.copy()
        target_data['target_options'] = {'name': 'SeriousDlqin2yrs'}
        target_data['target'] = {'name':target_data['target_options']['name']}
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            self.assertTrue(self.np_equal(ds.values,ds2.values), 'No changes should be made to ds')
        target_data['target_options'] = {'name': 'SeriousDlqin2yrs','positive_class':1}
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            self.assertTrue(self.np_equal(ds.values,ds2.values), 'No changes should be made to ds')

    def test_apply_target_options_float_binary_missing_vals_do_nothing(self):
        '''Use kickcars and IsBadBuy.  Don't provide a missing_maps_to value
        so that the missing values are ignored

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        testfile = 'kickcars-sample-200.csv'
        target_name = 'IsBadBuy'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file)
        target_data = self.workspace.data.copy()
        target_data['target_options'] = {'name': target_name, 'missing_maps_to':None}
        target_data['target'] = {'name':target_data['target_options']['name']}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        non_nulls = ds[target_name].notnull().sum()
        print ds[target_name]
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            print ds2[target_name]
            post_non_nulls = ds2[target_name].notnull().sum()
            self.assertGreater(non_nulls, 0, 'If this is zero, this is a pointless test')
            self.assertEqual(non_nulls, post_non_nulls, 'No mapping should have taken place')

    def test_apply_target_options_float_binary_missing_vals_make_zero(self):
        '''Use kickcars and IsBadBuy.  Make the missing values map to 0

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        testfile = 'kickcars-sample-200.csv'
        target_name = 'IsBadBuy'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file)
        target_data = self.workspace.data.copy()
        target_data['target_options'] = {'name': target_name, 'missing_maps_to':0.0}
        target_data['target'] = {'name': target_name, 'type':'Binary'}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        non_nulls = ds[target_name].notnull().sum()
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            post_nulls = ds2[target_name].isnull().sum()
            self.assertGreater(non_nulls, 0, 'If this is zero, this is a pointless test')
            self.assertEqual(0, post_nulls, 'There should be no nulls now (there are %d)' % post_nulls)

    def test_apply_target_options_cat_no_missing(self):
        '''Use allstate and Cat9.  There are only two values in this column, and
        no missing. The binary response mapping should be
        taken care of

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        testfile = 'allstate-nonzero-200.csv'
        target_name = 'Cat9'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file)
        target_data = self.workspace.data.copy()

        target_data['target_options'] = {'name': target_name}
        target_data['target'] = {'name':target_name, 'type':'Binary'}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        n_nulls = ds[target_name].isnull().sum()
        self.assertEqual(n_nulls, 0, 'This test case should have no nulls')
        self.assertEqual(ds[target_name].dtype, 'object', 'Want to check for success of map_binary_response')
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            self.assertEqual(ds2[target_name].dtype,float)
            for x in ds2[target_name]:
                self.assertIsInstance(x, numbers.Number)

    def test_apply_target_options_cat_missing_vars_do_nothing(self):
        '''Use fastiron and Stick.  There are lots of missing values; this test
        should ignore those rows entirely

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        testfile = 'fastiron-train-sample-small.csv'
        target_name = 'Stick'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file)
        target_data = self.workspace.data.copy()

        target_data['target_options'] = {'name': target_name, 'missing_maps_to':None}
        target_data['target'] = {'name':target_name, 'type':'Binary'}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        n_nulls = ds[target_name].isnull().sum()
        self.assertGreater(n_nulls, 0, 'This test case should have some nulls')
        self.assertEqual(ds[target_name].dtype, 'object', 'Want to check for success of CAT conversion')
        with patch.dict(self.workspace.data, target_data, clear=False):
            print self.workspace.data
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            print self.workspace.data
            for x in ds2[target_name]:
                self.assertIsInstance(x, numbers.Number)

    def test_apply_target_options_cat_missing_vars_make_choice(self):
        '''Use fastiron and Stick.  Map all the missing values to 'Standard'.
        '''
        testfile = 'fastiron-train-sample-small.csv'
        target_name = 'Stick'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file)
        target_data = self.workspace.data.copy()

        target_data['target_options'] = {'name': target_name, 'missing_maps_to':'Standard'}
        target_data['target'] = {'name':target_name, 'type':'Binary'}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        n_nulls = ds[target_name].isnull().sum()
        self.assertGreater(n_nulls, 0, 'This test case should have some nulls')
        self.assertEqual(ds[target_name].dtype, 'object', 'Want to check for success of CAT conversion')
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            for x in ds2[target_name]:
                self.assertIsInstance(x, numbers.Number)

    def test_apply_target_options_regression_no_missing(self):
        '''Use fastiron and SalePrice

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        testfile = 'fastiron-train-sample-small.csv'
        target_name = 'SalePrice'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file).copy()
        target_data = self.workspace.data.copy()

        target_data['target_options'] = {'name': target_name}
        target_data['target'] = {'name':target_name, 'type':'Regression'}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        n_nulls = ds[target_name].isnull().sum()
        self.assertEqual(n_nulls, 0, 'This test case should have no nans')
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            for i in ds2: # get_datframe may drop cols so check only output cols
                self.assertTrue(np.all(ds[i][ds[i].notnull()] == ds2[i][ds2[i].notnull()]), 'No changes should be made to ds')

    def test_apply_target_options_regression_missing_do_nothing(self):
        '''Use fastiron and SalePrice

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        testfile = 'fastiron-train-sample-small.csv'
        target_name = 'auctioneerID'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file)
        target_data = self.workspace.data.copy()

        target_data['target_options'] = {'name': target_name,'missing_maps_to':None}
        target_data['target'] = {'name':target_name, 'type':'Regression'}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        n_nulls = ds[target_name].isnull().sum()
        self.assertGreater(n_nulls, 0, 'This test case should have some nans')
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            post_n_nulls = ds2[target_name].isnull().sum()
            self.assertEqual(n_nulls, post_n_nulls, 'No values should have been imputed')

    def test_apply_target_options_regression_missing_make_zero(self):
        '''Use fastiron and SalePrice

        Simulate set_target by directly overwriting the self.workspace.data
        dictionary
        '''
        testfile = 'fastiron-train-sample-small.csv'
        target_name = 'auctioneerID'
        self.init_workspace(testfile)
        local_file = os.path.join(self.test_directory,testfile)
        storage.FileObject(testfile).get(local_file)
        ds = pandas.read_csv(local_file)
        target_data = self.workspace.data.copy()

        target_data['target_options'] = {'name': target_name, 'missing_maps_to':0.0}
        target_data['target'] = {'name':target_name, 'type':'Regression'}
        target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
        n_nulls = ds[target_name].isnull().sum()
        self.assertGreater(n_nulls, 0, 'This test case should have some nans')
        with patch.dict(self.workspace.data, target_data, clear=False):
            ds2 = self.workspace.get_dataframe(map_binary_response=True)
            post_n_nulls = ds2[target_name].isnull().sum()
            self.assertEqual(0, post_n_nulls, 'There should be no nans left')

    def test_apply_target_options_multiclass_positive_class(self):
        '''Use fastiron and Forks.  Choose the positive class to be 508

        This test will fail until we treat issue #289 because there are spaces
        in some of the categories for that column
        '''
        pass

    def test_getY(self):
        '''Make sure getY returns values in (0,1) for Classification tasks
        '''
        targets=['SeriousDlqin2yrs','IsBadBuy','Cat9']
        for i,testfile in enumerate(['credit-sample-200.csv', 'kickcars-sample-200.csv', 'allstate-nonzero-200.csv']):
            self.init_workspace(testfile)
            local_file = os.path.join(self.test_directory,testfile)
            storage.FileObject(testfile).get(local_file)
            ds = pandas.read_csv(local_file)
            target_data = self.workspace.data.copy()
            target_data['target_options'] = {'name': targets[i], 'missing_maps_to':0.0}
            target_data['target'] = {'name':targets[i], 'type':'Binary'}
            target_data['holdout_pct'] = 0.0 #this test assumes get_dataframe excludes no holdout data
            with patch.dict(self.workspace.data, target_data, clear=False):
                ds2 = self.workspace.get_dataframe(map_binary_response=True)
                y = getY(ds2,targets[i])
                self.assertSetEqual(set(y),{0,1})
                if testfile=='credit-sample-200.csv':
                    self.assertTrue(np.all(y.values==ds[targets[i]].values))
                else:
                    self.assertFalse(np.any(np.isnan(y.values.astype(float))))

    def test_validate_dataset(self):
        ws = self.workspace
        train_dataset_id = ws._get_filename()
        ds = ws.get_dataframe().copy()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        self.workspace.set_target(targetname)

        # test perfect match
        out = ws.validate_dataset(ds, train_dataset_id)
        self.assertIsInstance(out,ds.__class__)
        self.assertEqual(list(out.columns),list(ds.columns))
        self.assertEqual(out.shape,ds.shape)
        self.assertTrue(np.all(np.isfinite(out[targetname].values)))

    def test_validate_dataset_reshuffled(self):
        ws = self.workspace
        train_dataset_id = ws._get_filename()
        ds = ws.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        self.workspace.set_target(targetname)

        #test with reshuffled columns
        cols = range(ds.shape[1])
        random.shuffle(cols)
        dsnew = ds[cols]
        out = ws.validate_dataset(dsnew, train_dataset_id)
        self.assertIsInstance(out,ds.__class__)
        self.assertEqual(list(out.columns),list(ds.columns))
        self.assertEqual(out.shape,ds.shape)
        self.assertTrue(np.all(np.isfinite(out[targetname].values)))

    def test_validate_dataset_missing_response(self):
        ws = self.workspace
        train_dataset_id = ws._get_filename()
        ds = ws.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        self.workspace.set_target(targetname)

        #test missing response
        columns = list(ds.columns)
        target_position = columns.index(targetname)
        cols = [i for i in range(ds.shape[1]) if i != target_position]
        dsnew = ds[cols]
        out = ws.validate_dataset(dsnew, train_dataset_id)
        self.assertIsInstance(out,ds.__class__)
        self.assertEqual(list(out.columns),list(ds.columns))
        self.assertEqual(out.shape,ds.shape)
        self.assertTrue(np.all(np.isfinite(out[targetname].values)))

    def test_validate_dataset_missing_response_reshuffled(self):
        ws = self.workspace
        train_dataset_id = ws._get_filename()
        ds = ws.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        self.workspace.set_target(targetname)

        #test missing response and reshuffled columns
        columns = list(ds.columns)
        target_position = columns.index(targetname)
        cols = [i for i in range(ds.shape[1]) if i != target_position]
        random.shuffle(cols)
        dsnew = ds[cols]
        out = ws.validate_dataset(dsnew, train_dataset_id)
        self.assertIsInstance(out,ds.__class__)
        self.assertEqual(list(out.columns),list(ds.columns))
        self.assertEqual(out.shape,ds.shape)
        self.assertTrue(np.all(np.isfinite(out[targetname].values)))

    def test_validate_dataset_missing_response_column_match(self):
        ws = self.workspace
        train_dataset_id = ws._get_filename()
        ds = ws.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        self.workspace.set_target(targetname)

        #test column match but response has missing values
        dsnew = ds.copy()
        y = np.empty(dsnew.shape[0], dtype=float)
        y.fill(np.NaN)
        dsnew[targetname] = y
        out = ws.validate_dataset(dsnew, train_dataset_id)
        self.assertIsInstance(out,ds.__class__)
        self.assertEqual(list(out.columns),list(ds.columns))
        self.assertEqual(out.shape,ds.shape)
        self.assertTrue(np.all(np.isfinite(out[targetname].values)))

    def test_validate_dataset_shuffled_cols_nan_in_response(self):
        ws = self.workspace
        train_dataset_id = ws._get_filename()
        ds = ws.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        self.workspace.set_target(targetname)

        #test shuffled columns and response has missing values
        cols = range(ds.shape[1])
        random.shuffle(cols)
        dsnew = ds[cols]
        y = np.empty(dsnew.shape[0],dtype=float)
        y.fill(np.NaN)
        dsnew[targetname] = y
        out = ws.validate_dataset(dsnew, train_dataset_id)
        self.assertIsInstance(out,ds.__class__)
        self.assertEqual(list(out.columns),list(ds.columns))
        self.assertEqual(out.shape,ds.shape)
        self.assertTrue(np.all(np.isfinite(out[targetname].values)))

    def test_validate_dataset_raises(self):
        ws = self.workspace
        train_dataset_id = ws._get_filename()
        ds = ws.get_dataframe()
        #self.workspace.put_dataframe(ds)
        targetname = 'SeriousDlqin2yrs'
        self.workspace.set_target(targetname)
        #test other error
        dsnew = ds[range(5)]
        with self.assertRaises(ValueError):
            out = ws.validate_dataset(dsnew, train_dataset_id)

    def test_na(self):
        self.init_workspace('credit-na-test-200.csv')
        df = self.workspace.get_dataframe().copy()
        self.assertEqual(df['RevolvingUtilizationOfUnsecuredLines'].count(),186)
        self.assertEqual(df['RevolvingUtilizationOfUnsecuredLines'].dtype,'float64')

    def test_map_binary_response_smoke(self):
        """Test if binary responses are properly mapped smoke test. """
        with patch.object(self.workspace, 'get_target',
                          return_value={'name': 't', 'type': 'Binary'}) as mock_meth:
            df = pd.DataFrame({'t': ['yes', 'no', 'yes', 'no']})
            df_ = self.workspace.map_binary_response(df)
            np.testing.assert_array_equal(np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
                                          df_['t'].values)

    def test_map_binary_response_nan(self):
        """Test if binary responses are properly mapped; missing vals must be intakt. """
        with patch.object(self.workspace, 'get_target',
                          return_value={'name': 't', 'type': 'Binary'}) as mock_meth:
            df = pd.DataFrame({'t': ['yes', None, 'yes', 'no']})
            df_ = self.workspace.map_binary_response(df)
            np.testing.assert_array_equal(np.array([1.0, np.nan, 1.0, 0.0], dtype=np.float64),
                                          df_['t'].values)

            df = pd.DataFrame({'t': [1, None, 1, 0]})
            df_ = self.workspace.map_binary_response(df)
            np.testing.assert_array_equal(np.array([1.0, np.nan, 1.0, 0.0], dtype=np.float64),
                                          df_['t'].values)

    def test_map_binary_response_invalid(self):
        """Test if binary response mapping raises errors on malformed inputs. """
        with patch.object(self.workspace, 'get_target',
                          return_value={'name': 't', 'type': 'Binary'}) as mock_meth:

            # three labels - will raise ValueError
            df = pd.DataFrame({'t': [1, None, 1, 0, 'foo']})
            self.assertRaises(ValueError, self.workspace.map_binary_response, df)

            # one labels - will raise ValueError
            df = pd.DataFrame({'t': [1, None, 1, None, 1]})
            self.assertRaises(ValueError, self.workspace.map_binary_response, df)

    def test_rename_duplicate_names(self):
        new_list = ["a", "b", "c", "d"]
        existing_list = ["a", "b", "b 2", "c", "c 2", "c 3"]
        expected = ["a 2", "b 3", "c 4", "d"]
        renamed = self.workspace.rename_duplicate_feature_names(new_list, existing_list)
        self.assertEqual(renamed, expected)

if __name__ == '__main__':
    unittest.main()
