####################################################################
#
#       Test for MMApp Project Service Class
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc 2013
#
####################################################################

import unittest
import json
import numpy as np
from bson.objectid import ObjectId
from mock import patch, Mock, DEFAULT
import pytest
import json

from config.engine import EngConfig
from common.services.project import AimValidationException
from MMApp.entities.project import ProjectService, LEADERBOARD_TABLE
from MMApp.entities.project import SHARED_PREDICTION_API_KEYWORD
import MMApp.entities.project as project_module

from MMApp.task_descriptions import task_descriptions

from common.entities.blueprint import blueprint_id
from common.engine import metrics

# This import will overwrite the db_config used just about everywhere
# but, since this is the testing environment, we want all the tests
# to use test DBs, so that's fine.
from config.test_config import db_config
from common.wrappers import database
import common.services.eda


class ProjectServiceTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempstore = database.new('tempstore')
        self.persistent = database.new('persistent')

    @classmethod
    def tearDownClass(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table=LEADERBOARD_TABLE)
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='predictions')
        self.persistent.destroy(table='predict_parts')

    def setUp(self):
        self.tempstore.conn.flushdb()
        self.persistent.destroy(table=LEADERBOARD_TABLE)
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='predictions')
        self.persistent.destroy(table='predict_parts')

        self.uid = ObjectId(None)
        self.permissions = {str(self.uid): {'CAN_VIEW': True}}
        self.data = {'test': 'test',
                     'permissions': self.permissions,
                     'target': {'name': 'TargetName'}}
        self.pid = self.persistent.create(table='project', values=self.data)
        self.project_service = ProjectService(self.pid, self.uid)


        self.addCleanup(self.stopPatching)

        permission_patcher = patch("MMApp.app.ProjectService.has_permission")
        permission_patcher.start()
        permission_patcher.return_value = True

        self.patchers = []
        self.patchers.append(permission_patcher)

    def stopPatching(self):
        super(ProjectServiceTestCase, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    @pytest.mark.unit
    def test_get_auth_data_does_not_change_eng_config(self):
        serialized_config = json.dumps(EngConfig)

        self.project_service.get_auth_data()

        after_the_fact_config = json.dumps(EngConfig)

        self.assertEqual(serialized_config, after_the_fact_config, 'EngConfig should remain immutable')

    @pytest.mark.db
    def test_create_leaderboard_item(self):
        service = ProjectService()
        pid = '5239fff2637aba042f4680f7'
        uid = '5239fc3d637aba7f1a53ed2f'
        lid = service.create_leaderboard_item({'name': 'new_leaderboard_item', 'pid' : pid, 'uid':uid})
        self.assertIsInstance(lid, str)

        leaderboard = self.persistent.read(table=LEADERBOARD_TABLE,
                                           condition={'_id':ObjectId(lid),
                                                      'pid':ObjectId(pid),
                                                      'uid':ObjectId(uid)},
                                           result=[])[0]
        self.assertIsNotNone(leaderboard)

    def test_delete_leaderboard_items(self):
        pid = ObjectId()
        ids = []
        for i in range(5):
            lid = self.persistent.create(table=LEADERBOARD_TABLE, values = {'pid': pid})
            ids.append(lid)

        items = self.persistent.read(table=LEADERBOARD_TABLE,
                condition= {'_id' : {'$in': ids}}, result = [])

        self.assertEqual(len(items), 5)

        service = ProjectService(pid, ObjectId())
        service.delete_leaderboard_items(ids)

        items = self.persistent.read(table=LEADERBOARD_TABLE,
                condition= {'_id' : {'$in': ids}}, result = [])

        self.assertFalse(items)

    @pytest.mark.db
    @patch('MMApp.entities.project.EdaService')
    def test_get_leaderboard_item(self, MockEdaService):
        #Insert a record directly into the dabase. This helps making this test more independent. However, ideally database access should be mocked.
        mock_eda = MockEdaService.return_value
        mock_eda.get_map.return_value = {
            'TargetName': {'metric_options': {'all': [
                {'short_name': 'MAD',
                 'description': 'Mothers Against Drunk'}]}}}

        bp = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GLMB'],'P']}
        new_leaderboard_item = {'pid' : self.pid, 'uid':self.uid,
                                'key1' : 'value1', 'holdout':'asdf',
                                'blueprint':bp, 'blueprint_id':'BPKEY',
                                'test': {'metrics': ['Gini', 'MAD'],
                                         'Gini': [0.1],
                                         'MAD': [0.5]}
                                }
        new_project_menu_item = {'menu': {'BPKEY': {
            'diagram': '{"taskMap":{"fake":{"label":"also_fake"}}}'}},
                                 'pid': self.pid}
        lid = self.persistent.create(table=LEADERBOARD_TABLE,
                                      values=new_leaderboard_item)
        lid = str(lid)
        new_metablueprint_menu = self.persistent.create(table='metablueprint',
                values=new_project_menu_item)

        #locked holdout
        saved_record = self.project_service.get_leaderboard_item(lid)
        self.assertLessEqual(set(saved_record.keys()),
                             set(new_leaderboard_item.keys()))
        self.assertNotIn('Gini', saved_record['test']['metrics'])
        self.assertTrue('holdout' not in saved_record)

        #bpData but no holdout
        saved_record = self.project_service.get_leaderboard_item(lid, bpData=True)
        self.assertTrue('holdout' not in saved_record)
        self.assertTrue('bpData' in saved_record)
        self.assertTrue('blueprint' not in saved_record)

        #unlocked holdout
        self.project_service.unlock_holdout()

        saved_record = self.project_service.get_leaderboard_item(lid)
        self.assertLessEqual(set(saved_record.keys()),
                             set(new_leaderboard_item.keys()))
        self.assertTrue('holdout' in saved_record)

        #bpData and holdout
        saved_record = self.project_service.get_leaderboard_item(lid, bpData=True)
        self.assertTrue('holdout' in saved_record)
        self.assertTrue('bpData' in saved_record)
        self.assertTrue('blueprint' not in saved_record)

    def test_get_censored_fields(self):
        fields = self.project_service.get_censored_fields([])
        self.assertTrue(fields)

        exclude = self.project_service.ALLOWED_KEYS_FOR_UI
        fields = self.project_service.get_censored_fields(exclude)
        self.assertFalse(fields)

    @pytest.mark.db
    def test_save_leaderboard_item(self):
        service = ProjectService()

        pid = '5239fc3d637aba7f1a53ed2f'
        uid = '5239fff2637aba042f4680f7'
        leaderboard_item = {'key1' : 'value1', 'pid' : pid, 'uid':uid}
        l_id = service.create_leaderboard_item(leaderboard_item)
        l_id = str(l_id)

        leaderboard_item['key1'] = 'value2'
        service.save_leaderboard_item(l_id, leaderboard_item)

        saved_record = self.persistent.read(table=LEADERBOARD_TABLE,
                                            condition={'_id':ObjectId(l_id)},
                                            result=[])[0]
        self.assertDictContainsSubset(saved_record, leaderboard_item)

    @pytest.mark.db
    def test_set_autopilot_mode(self):
        self.project_service.set_autopilot_mode(1)
        query = self.persistent.read(table='project',
                                     condition={'_id':self.pid},
                                     result = [])
        query = query[0]
        self.data['mode'] = 1
        self.assertEqual(query, self.data)

    @pytest.mark.db
    def test_set_project_stage(self):
        #invalid stage value
        self.assertRaises(ValueError,self.project_service.set_project_stage,'asdf')
        check = self.tempstore.read(keyname='stage', index=str(self.pid))
        self.assertEqual(check,'')
        #valid stage value
        self.project_service.set_project_stage('eda:1234')
        check = self.tempstore.read(keyname='stage',
                                    index=str(self.pid))
        self.assertEqual(check,'eda:1234')

    @pytest.mark.db
    def test_get_leaderboard(self):
        #no values on mongodb
        check = self.project_service.get_leaderboard()
        self.assertEqual(check,[])
        #valid values on mongodb
        self.persistent.create(table=LEADERBOARD_TABLE,
                values={'pid':self.pid,'test':'a','holdout':'asdf','ilegal_key':'asdf'})
        self.persistent.create(table=LEADERBOARD_TABLE,
                values={'pid':self.pid,'test':'b','holdout':'sdfg','ilegal_key':'asdf'})
        self.persistent.create(table=LEADERBOARD_TABLE,
                values={'test':'c'})
        check = self.project_service.get_leaderboard()
        self.assertEqual(len(check),2)
        test_values = [item['test'] for item in check]
        self.assertEqual(set(test_values),set(['a','b']))
        for item in check:
            self.assertEqual(item['pid'],self.pid)
            self.assertEqual(set(item.keys()),set(['_id','pid','test','ilegal_key']))

        #censoring
        check = self.project_service.get_leaderboard(UI_censoring=True)
        self.assertEqual(len(check),2)
        for item in check:
            self.assertEqual(item['pid'],self.pid)
            self.assertEqual(set(item.keys()),set(['_id','pid','test']))

        #unlocked holdout
        self.project_service.unlock_holdout()
        check = self.project_service.get_leaderboard()
        self.assertEqual(len(check),2)
        for item in check:
            self.assertEqual(item['pid'],self.pid)
            self.assertEqual(set(item.keys()),set(['_id','pid','test','holdout','ilegal_key']))

        #censoring
        check = self.project_service.get_leaderboard(UI_censoring=True)
        self.assertEqual(len(check),2)
        for item in check:
            self.assertEqual(item['pid'],self.pid)
            self.assertEqual(set(item.keys()),set(['_id','pid','test','holdout']))

    @patch('MMApp.entities.project.EdaService', autospec=True)
    def test_get_leaderboard_only_returns_valid_metrics(self, MockEdaService):
        mock_eda_service = MockEdaService.return_value
        mock_eda_service.get_map.return_value = {
                'TargetName': {'metric_options': {'all': [
                    {'short_name': 'MAD',
                     'description': 'Mothers Against Drunk'}]}}}
        for i in range(4):
            self.persistent.create(table=LEADERBOARD_TABLE,
                                   values={'pid':self.pid,
                                           'test': {'metrics': ['MAD', 'Gini'],
                                                    'MAD': [0.5],
                                                    'Gini': [0.5]},
                                           'holdout':'asdf'})

        leaderboard = self.project_service.get_leaderboard()
        self.assertGreater(len(leaderboard), 0)
        for item in leaderboard:
            self.assertTrue(all([key in ['MAD', 'metrics']
                                  for key in item['test'].keys()]))

    @pytest.mark.db
    def test_get_leaderboard_specific_fields(self):
        bunch_of_keys = [u'features', u'best_parameters', u'qid',
                         u'task_version', u'part_size', u'max_reps',
                         u'holdout_scoring_time', u'task_info', u'ec2', u'uid',
                         u'dataset_id', u'time_real', u'holdout',
                         u'grid_scores', u'lid', u'resource_summary',
                         u'training_dataset_id', u'icons', u'total_size',
                         u'partition_stats', u'job_info', u'parts', u'test',
                         u'parts_label', u'blueprint', u'blueprint_id',
                         u'lift', u'bp', u'task_parameters', u'task_cnt',
                         u'max_folds', u'metablueprint', u'insights',
                         u'samplepct', u'training_dataset_name',
                         u'finish_time', u'vertices', u'holdout_size', u's',
                         u'extras', u'reference_model', u'all_parameters',
                         u'time', u'model_type', u'vertex_cnt',
                         u'blend', u'dataset_name']

        values = {key: key for key in bunch_of_keys}
        values['pid'] = self.pid
        self.persistent.create(table=LEADERBOARD_TABLE,
                               values=values)

        desired_fields = ['blueprint_id', 'blueprint', 'blender', 'bp',
                          'dataset_name', 'samplepct']
        ldb = self.project_service.get_leaderboard(fields=desired_fields)
        model = ldb[0]
        self.assertNotIn('blender', model)  # Won't create nonexistent values
        self.assertIn('_id', model) # _id always comes back
        self.assertEqual(set(model.keys()),
                         {'_id', 'blueprint_id', 'blueprint', 'bp',
                          'dataset_name', 'samplepct'})

    @pytest.mark.db
    def test_get_leaderboard_holdout(self):
        #no values on mongodb
        check = self.project_service.get_leaderboard_holdout()
        self.assertEqual(check,[])
        #valid values on mongodb
        self.persistent.create(table=LEADERBOARD_TABLE,
                               values={'pid':self.pid,'test':'a','holdout':'asdf'})
        self.persistent.create(table=LEADERBOARD_TABLE,
                               values={'pid':self.pid,'test':'b','holdout':'sdfg'})
        self.persistent.create(table=LEADERBOARD_TABLE,
                               values={'test':'c'})
        check = self.project_service.get_leaderboard_holdout()
        self.assertEqual(len(check),2)
        for item in check:
            self.assertEqual(set(item.keys()),set(['_id','holdout']))

    @pytest.mark.db
    def test_get_leaderboard_gets_all(self):
        '''The default read from mongo limits to 10 items.  We need to make sure
        that we are getting all.'''
        ldbvalues = [{'pid':self.pid,
                      'test': {'metrics': ['MAD', 'Gini'],
                               'MAD': [0.1 * i],
                               'Gini': [0.01 * i]},
                      'holdout':str(i)} for i in xrange(15)]
        for values in ldbvalues:
            self.persistent.create(table=LEADERBOARD_TABLE, values=values)
        check = self.project_service.get_leaderboard()
        self.assertEqual(len(check), 15)

    @pytest.mark.db
    def test_save_leaderboard_item(self):
        #update an item that exists in the database
        lid = self.persistent.create(table=LEADERBOARD_TABLE,
                                     values={'test':'test'})
        query = self.persistent.read(table=LEADERBOARD_TABLE,
                                     condition={'_id':lid},
                                     result = [])[0]
        self.assertEqual(query,{'test':'test','_id':lid})
        self.project_service.save_leaderboard_item(lid,{'update':'update'})
        query = self.persistent.read(table=LEADERBOARD_TABLE,
                                     condition={'_id':lid},
                                     result=[])[0]
        self.assertEqual(query,{'test':'test','update':'update','_id':lid})
        #try to update an item that doesn't exist
        # Not working due to the way that mongo wrapper wraps update
        #self.project_service.save_leaderboard_item(self.pid,{'update':'update'})
        #query = self.persistent.read(table=LEADERBOARD_TABLE,
        #                             condition={'_id':self.pid},
        #                             result=[])
        #self.assertEqual(query,[])

    @pytest.mark.db
    def test_get_invalid_prediction(self):
        #try to get an item that doesn't exist
        check = self.project_service.get_predictions(self.pid,self.pid)
        self.assertEqual(check,None)

    @pytest.mark.db
    def test_large_prediction(self):
        #get valid item
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values = {'pid' : self.pid})
        dataset_id = '5239fc3d637aba7f1a53ed2f'

        # create prediction > 50K rows
        prediction1 = {'test':'x','lid':lid,'dataset_id':dataset_id, 'pid' : self.pid, 'row_index': np.ones(60000).tolist() }
        item = {'lid':lid,'dataset_id':dataset_id }
        # store large prediction
        pred_id = self.project_service.save_predictions(prediction1,item)
        prediction1['_id'] = pred_id
        # prediction should be split into 2 pieces
        self.assertEqual(len(prediction1['pieces']),2)
        # large list should be removed from original doc
        self.assertTrue('row_index' not in prediction1)
        # read back prediction and check size
        actual_prediction = self.project_service.get_predictions(lid,dataset_id)
        self.assertEqual(len(actual_prediction['row_index']),60000)
        # check if first piece is 50K
        piece = prediction1['pieces'][0]
        query = self.persistent.read(table='predict_parts',
                                     condition={'_id':piece},
                                     result=[])[0]
        # first piece should be 50000 rows (the default in save_predictions)
        self.assertEqual(len(query['row_index']),50000)
        # for the sake of speed just check the first item
        self.assertEqual(query['row_index'][0],1)

        # test saving prediction updates the original
        prediction2 = {'test':'x','lid':lid,'dataset_id':dataset_id, 'pid' : self.pid, 'row_index': np.zeros(60000).tolist() }
        new_pred_id = self.project_service.save_predictions(prediction2,item)
        # check result for update
        self.assertTrue(new_pred_id['updatedExisting'])
        query = self.persistent.read(table='predictions',
                                     condition={'_id':pred_id['upserted']},
                                     result=[])[0]
        # ids of pieces should be the same
        self.assertEqual(prediction1['pieces'],query['pieces'])
        piece = prediction1['pieces'][0]
        query = self.persistent.read(table='predict_parts',
                                     condition={'_id':ObjectId(piece)},
                                     result=[])[0]
        # pieces should now have 0s instead of 1s
        self.assertEqual(query['row_index'][0],0)

    @pytest.mark.db
    def test_delete_large_prediction(self):
        #get valid item
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values = {'pid' : self.pid})
        dataset_id = '5239fc3d637aba7f1a53ed2f'

        # create prediction > 50K rows
        prediction1 = {'test':'x','lid':lid,'dataset_id':dataset_id, 'pid' : self.pid, 'row_index': np.ones(60000).tolist() }
        item = {'lid':lid,'dataset_id':dataset_id }

        # store large prediction
        pred_id = self.project_service.save_predictions(prediction1, item)

        # delete prediction
        self.project_service.delete_predictions(str(lid), dataset_id = ProjectService.ALL_DATASETS)

        # check db
        check1 = self.persistent.read(table='predictions', condition={'lid':lid}, result=[])
        check2 = self.persistent.read(table='predict_parts', result=[])
        self.assertEqual(check1, [])
        self.assertEqual(check2, [])

        # try to read prediction
        prediction = self.project_service.get_predictions(lid, dataset_id)
        self.assertEqual(prediction, None)

    @pytest.mark.db
    def test_delete_multiple_predictions(self):
        dataset_id = self.persistent.create(table='metadata')
        dataset_id = str(dataset_id)

        lid1 = self.persistent.create(table=LEADERBOARD_TABLE, values = {'pid' : self.pid})
        lid2 = self.persistent.create(table=LEADERBOARD_TABLE, values = {'pid' : self.pid})

        prediction_1 = self.persistent.create(table='predictions',
            values = {'dataset_id': dataset_id, 'lid' : lid1, 'pid': self.pid})
        prediction_2 = self.persistent.create(table='predictions',
            values = {'dataset_id': dataset_id, 'lid' : lid2, 'pid': self.pid})

        deleted = self.project_service.delete_predictions(leaderboard_id = ProjectService.ALL_MODELS,
            dataset_id = dataset_id)

        self.assertEqual(deleted, 2)

        predictions = self.persistent.read(table = 'predictions', result = [],
            condition = {'_id' : {'$in' : [prediction_1, prediction_2]}})

        self.assertFalse(predictions)

    @pytest.mark.db
    def test_get_valid_prediction(self):
        #get valid item
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values = {'pid' : self.pid})
        dataset_id = '5239fc3d637aba7f1a53ed2f'

        prediction1 = {'test':'x','lid':lid,'dataset_id':dataset_id, 'pid' : self.pid}
        oid1 = self.persistent.create(table='predictions',
                                      values=prediction1)
        prediction1['_id'] = oid1

        actual_prediction = self.project_service.get_predictions(lid,dataset_id)

        self.assertEqual(actual_prediction,prediction1)

    @pytest.mark.db
    def test_get_predictions_works_with_no_pid(self):
        #get valid item
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values = {'pid' : self.pid})
        dataset_id = '5239fc3d637aba7f1a53ed2f'

        expected_prediction = {'test':'x','lid':lid,'dataset_id':dataset_id}

        oid1 = self.persistent.create(table='predictions',
                                      values=expected_prediction)

        expected_prediction['_id'] = oid1

        actual_prediction = self.project_service.get_predictions(lid,dataset_id)

        self.assertIn('pid', actual_prediction)
        self.assertEqual(self.project_service.pid, self.pid)
        self.assertEqual(actual_prediction['pid'],self.pid)
        self.assertEqual(actual_prediction['lid'], lid)
        self.assertEqual(actual_prediction['dataset_id'], dataset_id)

    @pytest.mark.db
    def test_get_project_list(self):
        #empty self.project_service list
        project_service = ProjectService()
        query = project_service.get_project_list(ObjectId(None))
        self.assertEqual(query,[])
        #one self.project_service on list
        testid = self.persistent.create(table='project',
                                        values={'test':'test'})
        uid = self.persistent.create(table='users',
                                     values={'roles':
                                             {str(testid): ['OWNER']}})
        self.persistent.update(table='project',
                               condition={'_id': testid},
                               values={'uid': str(uid),
                                       'roles': {str(uid): ['OWNER']}})
        project_list = project_service.get_project_list(str(uid))

        self.assertEqual(len(project_list),1)
        project = project_list[0]

        def check_all_keys(project_record):
            check = {
                'project_name': 'Untitled Project',
                'created': None,
                'originalName': None,
                'mode': 1,
                'active': None,
                '_id': testid,
                'stage': '',
                'target': None,
                'metric': None,
                'holdout_unlocked': None
            }

            self.assertTrue(all(k in project for k in check),
                'Project keys {0} did not include the required keys: {1}'.format(project.keys(), check.keys()))

        check_all_keys(project)

        #two projects on list
        testid2 = self.persistent.create(table='project',
                                         values={'test':'test'})
        self.persistent.update(table='project',
                               condition={'_id': testid2},
                               values={'uid': str(uid),
                                       'roles': {str(uid): ['OWNER']}})
        self.persistent.update(table='users',
                               condition={'_id': uid},
                               values={'roles': {str(testid): ['OWNER'],
                                                 str(testid2): ['OWNER']}})
        project_list = project_service.get_project_list(str(uid))
        self.assertEqual(len(project_list),2)

        for project in project_list:
            check_all_keys(project)

        # Make sure deleted projects don't show up in the list
        project_service.mark_as_deleted(testid)
        project_list = project_service.get_project_list(str(uid))
        self.assertEqual(len(project_list), 1)

        project_service.mark_as_deleted(testid2)
        project_list = project_service.get_project_list(str(uid))
        self.assertEqual(len(project_list), 0)

    @pytest.mark.db
    @patch('common.services.eda.assert_valid_id')
    @patch.object(common.services.eda.EdaService, 'assert_has_permission')
    def test_get_project_data(self,vmock,pmock):
        #no project data in Redis or Mongo
        self.pid = self.persistent.create(table='unitTesting',
                                     values={'test':'test'})
        project = ProjectService(self.pid)
        self.assertRaises(ValueError, project.get_project_data)
        #data in Mongo
        dataset_id = '52b6121e076329407cb2c88c'
        pid = self.persistent.create(table='project',
                                     values={'default_dataset_id':dataset_id,'stage':'aim'})
        self.persistent.update(table='metadata',
                               values={'pid':pid,'dataset_id':dataset_id,'originalName':'asdf','shape':'asdf','created':'asdf','columns':'asdf'},
                               condition={"_id": ObjectId(dataset_id)})
        es = common.services.eda.EdaService(pid,self.uid,dataset_id)
        es.update({'asdf':{'mssm':'blah'}})

        project = ProjectService(pid)
        query = project.get_project_data()

        check = {'eda': {}, 'eda_labels1': [], 'eda_labels0': [], 'shape': u'asdf', 'project_name': 'Untitled Project',
                'active': 1, 'univariate': {}, 'stage': u'aim', 'created': u'asdf', 'filename': u'asdf',
                '_id': pid, 'columns': u'asdf'}
        self.assertEqual(check, query)

        es.update({'univariate':{'asdf':'blah'}})
        query = project.get_project_data()
        check = {'eda': {}, 'eda_labels1': [], 'eda_labels0': [], 'shape': u'asdf', 'project_name': 'Untitled Project',
                'active': 1, 'univariate': {}, 'stage': u'aim', 'created': u'asdf', 'filename': u'asdf',
                '_id': pid, 'columns': u'asdf'}
        self.assertEqual(check, query)

    @pytest.mark.db
    @patch('MMApp.entities.project.database.ObjectId')
    def test_get_all_metadata_with_length_over_default_limit(self, fake_OID):
        pid = '1234'
        dataset_count = 12
        for dataset_id in map(str, range(dataset_count)):
            self.persistent.create(table='metadata',
                                   values={'pid': pid,
                                           'dataset_id': dataset_id,
                                           'originalName': 'asdf'+dataset_id,
                                           'shape': 'asdf'+dataset_id,
                                           'created': 'asdf'+dataset_id,
                                           'columns': 'asdf'+dataset_id,
                                           'newdata': True})
        fake_OID.return_value = pid
        project = ProjectService()
        project.pid = pid
        metadata = project.get_all_metadata()
        self.assertEqual(len(metadata), dataset_count)

    @pytest.mark.db
    @patch('MMApp.entities.project.database.ObjectId')
    def test_get_all_metadata_failure(self, fake_OID):
        pid = '1234'
        self.persistent.destroy(table='metadata',
                                condition={'pid':pid})
        fake_OID.return_value = pid
        project = ProjectService()
        project.pid = pid
        self.assertRaises(ValueError, project.get_all_metadata)

    @pytest.mark.db
    #@patch('MMApp.entities.project.database.ObjectId')
    @patch('common.services.eda.assert_valid_id')
    @patch.object(common.services.eda.EdaService,'assert_has_permission')
    def test_validate_target_with_cat_variable_no_err(self, fake_VID, fake_hp):
        pid = '5239fff2637aba042f4680f6'
        dataset_id = '5239fc3d637aba7f1a53ed2f'
        project = ProjectService()
        project.pid = pid
        project.uid = self.uid
        try:
            self.persistent.destroy(table='eda', condition={'dataset_id':dataset_id, 'pid':pid})
            self.persistent.destroy(table='eda_map', condition={'dataset_id':dataset_id, 'pid':pid})
        except:
            pass
        es = common.services.eda.EdaService(pid,self.uid)
        es.update(
                       {'a':
                           {'profile':
                               {'plot':
                                   [('A','1'),('B','2')],
                                'type': 'N',
                               },
                             'summary':(2,0),
                           }
                       }
                   )

        with patch.object(project, 'get_metadata') as mock_meta:
            mock_meta.return_value = {'columns':[[0, 'a', 0]],
                                      'varTypeString':'C'}
            target = 'a'
            project.validate_target(target, dataset_id)

    @pytest.mark.db
    def test_get_name_no_name_present(self):
        pid = self.persistent.create(table='project', values={'nothing':'here'})
        p_service = ProjectService()
        p_service.pid = pid

        result = p_service.name
        self.assertEqual(result, 'Unnamed Project')

    @pytest.mark.db
    def test_get_name_name_present(self):
        pid = self.persistent.create(table='project',
                                     values={'project_name':'A Named Project'})

        p_service = ProjectService(pid)
        result = p_service.name
        self.assertEqual(result, 'A Named Project')

    @pytest.mark.db
    def test_set_name(self):
        pid = '1234'
        p_service = ProjectService()
        p_service.pid = pid

        p_service.name = 'A new name'
        result = p_service.name
        self.assertEqual(result, 'A new name')

    @pytest.mark.db
    def test_get_metric_none_defined(self):
        pid = self.persistent.create(table='project', values={'nothing':'here'})
        p_service = ProjectService()
        p_service.pid = pid

        result = p_service.metric
        self.assertEqual(result, 'Undefined')

    @pytest.mark.db
    def test_get_metric_when_present(self):
        pid = self.persistent.create(table='project',
                                     values={'metric':'RMSLE'})

        p_service = ProjectService(pid)
        result = p_service.metric
        self.assertEqual(result, 'RMSLE')

    @pytest.mark.db
    def test_set_metric(self):
        pid = '1234'
        p_service = ProjectService()
        p_service.pid = pid

        p_service.metric = metrics.MAD
        result = p_service.metric
        self.assertEqual(result, metrics.MAD)

        # now check if we raise error on unsupported metric
        def setter(m):
            p_service.metric = m
        self.assertRaises(ValueError, setter, 'MAE')


    @pytest.mark.unit
    def test_get_users_from_project(self):
        service = ProjectService()

        project = {
            '_id' : ObjectId('52f17fde637aba4b62592698'),
            'roles' : {
                '52dc3112637aba73d06e48c8' : ['OWNER'],
                '5239fc3d637aba7f1a53ed2f' : ['OBSERVER']
            }
        }

        with patch.object(service, 'persistent') as mock_db:
            mock_db.read.return_value = [
                { 'username': 'account@datarobot.com', '_id' : '5239fc3d637aba7f1a53ed2f'},
                { 'username': '810dc61f-7770-42bd-a487-725cb712939a', '_id' : '52dc3112637aba73d06e48c8'}
            ]
            users = service.get_users_from_project(project)

            self.assertIsInstance(users, list)
            self.assertEqual(len(users), 2)

            while True:
                if not users:
                    break

                user = users.pop()

                user_keys = ['username', '_id', 'roles']
                self.assertTrue(all([k in user for k in user_keys]))

                uid = user['_id']
                self.assertItemsEqual(project['roles'][uid], user['roles'])

    @pytest.mark.unit
    @patch('MMApp.entities.project.RoleProvider', autospec = True)
    def test_set_role_for_team_member_with_no_permissions(self, MockRoleProvider):
        service = ProjectService()

        role_provider_instance = MockRoleProvider.return_value
        role_provider_instance.has_permission.return_value = False

        self.assertRaisesRegexp(ValueError, 'does not have permissions', service.set_role_for_team_member, None, None)

    @pytest.mark.unit
    @patch('MMApp.entities.project.RoleProvider', autospec = True)
    def test_set_role_for_team_member(self, MockRoleProvider):
        service = ProjectService()
        team_member_uid = '52dc081379cbafddb41ca40e'
        roles = ['OBSERVER']

        role_provider_instance = MockRoleProvider.return_value
        role_provider_instance.has_permission.return_value = True
        role_provider_instance.get_uids_by_permission.return_value = ['52dc081379cbafddb41ca40e', '52ed139772307c28e3bf496a']

        result = service.set_role_for_team_member(team_member_uid, roles)
        self.assertTrue(result)

    @pytest.mark.unit
    @patch('MMApp.entities.project.RoleProvider', autospec = True)
    def test_set_role_for_team_member_with_no_owner(self, MockRoleProvider):
        service = ProjectService()
        owner_id = team_member_uid = '111111111111111111111111'
        roles = [u'OBSERVER']

        role_provider_instance = MockRoleProvider.return_value
        role_provider_instance.has_permission.return_value = True
        role_provider_instance.get_uids_by_roles.return_value = [owner_id]

        self.assertRaisesRegexp(ValueError, 'must have at least one owner', service.set_role_for_team_member, team_member_uid, roles)

    @pytest.mark.unit
    @patch('MMApp.entities.project.RoleProvider', autospec = True)
    def test_remove_from_project(self, MockRoleProvider):
        service = ProjectService()
        team_member_uid = '111111111111111111111111'

        role_provider_instance = MockRoleProvider.return_value
        role_provider_instance.delete_roles.return_value = True

        result = service.remove_from_project(team_member_uid)
        self.assertTrue(result)

    @pytest.mark.unit
    @patch('MMApp.entities.project.RoleProvider', autospec = True)
    def test_remove_owner_from_project(self, MockRoleProvider):
        service = ProjectService()
        owner_uid = team_member_uid = '111111111111111111111111'

        role_provider_instance = MockRoleProvider.return_value
        role_provider_instance.get_uids_by_roles.return_value = [owner_uid]
        role_provider_instance.delete_roles.return_value = True


        self.assertRaisesRegexp(ValueError, 'must have at least one owner', service.remove_from_project, team_member_uid)

    @pytest.mark.db
    def test_same_token(self):
        token1 = self.project_service.get_token()
        token2 = self.project_service.get_token()
        self.assertEqual(token1, token2)


    @pytest.mark.db
    def test_submitted_target(self):
        target = 'target-x'
        self.project_service.set_submitted_target(target)
        actual = self.project_service.get_submitted_target()

        self.assertEqual(target, actual)

        self.project_service.clear_submitted_target()

        project = self.project_service.get()

        self.assertNotIn('submitted_target', project)

    def test_set_aim_defaults(self):
        # validate basic/old eda aim post
        postdata = {'pid' : '1'}
        expected = {'cv_method':'RandomCV','reps':5,'folds':5,'holdout_pct':20,'ui_validation_type':'CV'}
        validated_data = self.project_service.validate_cv_method(postdata)
        self.assertEqual(validated_data, expected)

    def test_validate_commonCV(self):
        # Test proper submission of CVH
        postdata = {'pid':'1','cv_method':"RandomCV",'validation_type':'CV','reps':10,'holdout_pct':30}
        expected = {'cv_method':'RandomCV','reps':10,'holdout_pct':30,'ui_validation_type':'CV'}
        self.assertEqual(self.project_service.validate_cv_method(postdata), expected)

        # Test case of both reps and validation_pct being submitted
        postdata = {'pid':'1','cv_method':"RandomCV",'validation_type':'CV','reps':10,'holdout_pct':30,'validation_pct':30}
        self.assertRaisesRegexp(AimValidationException,
            'Cannot submit both reps and validation percent', self.project_service.validate_cv_method, postdata)

        # Test case of reps over threshold
        postdata = {'pid':'1','cv_method':"StratifiedCV",'validation_type':'CV','reps':1000000,'holdout_pct':30}
        self.assertRaisesRegexp(AimValidationException,
            "Invalid number of CV folds \(2 - 999999\)", self.project_service.validate_cv_method, postdata)

        # Test case of bad reps
        postdata = {'pid':'1','cv_method':"StratifiedCV",'validation_type':'CV','reps':'dog','holdout_pct':30}
        self.assertRaisesRegexp(AimValidationException,
            "Invalid number of CV folds \(2 - 999999\)", self.project_service.validate_cv_method, postdata)

        # Test proper submission of TVH
        postdata = {'pid':'1','cv_method':"StratifiedCV",'validation_type':'TVH','validation_pct':60,'holdout_pct':40}
        expected = {'reps':1,'validation_pct':0.6,'holdout_pct':40,'cv_method':'StratifiedCV','ui_validation_type':'TVH'}
        self.assertEqual(self.project_service.validate_cv_method(postdata), expected)

        # Test case of both validation_pct over threshold
        postdata = {'pid':'1','cv_method':"RandomCV",'validation_type':'TVH','validation_pct':100,'holdout_pct':40}
        self.assertRaisesRegexp(AimValidationException,
            'Invalid number for validation percentage \(1-99\)%', self.project_service.validate_cv_method, postdata)

        # Test case of no holdout_pct
        postdata = {'pid':'1','cv_method':"RandomCV",'validation_type':'TVH','validation_pct':60}
        self.assertRaisesRegexp(AimValidationException,
            'Invalid number for holdout percentage \(0-98\)%', self.project_service.validate_cv_method, postdata)

    def test_validate_userCV(self):
        # Test improper submission of cross-validation:
        postdata = {'pid':'1','cv_method':"UserCV",'validation_type':'TVH','training_level':'TLVL','validation_level':'VLVL','holdout_level':'HLVL'}
        self.assertRaisesRegexp(AimValidationException,
            'partition_col key missing for partition type: UserCV', self.project_service.validate_cv_method, postdata)

        # Test proper submission of cross-validation:
        postdata = {'pid':'1','cv_method':"UserCV",'validation_type':'TVH','partition_col':'USER','cv_holdout_level':'HLVL','training_level':'TLVL'}
        expected = {'partition_col':'USER','cv_holdout_level':'HLVL','cv_method':'UserCV','ui_validation_type':'TVH'}
        self.assertEqual(self.project_service.validate_cv_method(postdata), expected)

        # Test proper submission of train/test/holdout split:
        postdata = {'pid':'1','cv_method':"UserCV",'validation_type':'TVH','partition_col':'USER','training_level':'TLVL','validation_level':'VLVL','holdout_level':'HLVL'}
        expected = {'partition_col':'USER','training_level':'TLVL','validation_level':'VLVL','holdout_level':'HLVL','cv_method':'UserCV','ui_validation_type':'TVH'}
        self.assertEqual(self.project_service.validate_cv_method(postdata), expected)

        # Test missing keys submission of train/test/holdout split:
        postdata = {'pid':'1','cv_method':"UserCV",'validation_type':'TVH','partition_col':'USER','training_level':'TLVL','holdout_level':'HLVL'}
        self.assertRaisesRegexp(AimValidationException,
            'keys missing for partition type: UserCV', self.project_service.validate_cv_method, postdata)

    def test_validate_groupCV(self):
        # Test improper submission:
        postdata = {'pid':'1','cv_method':"GroupCV",'validation_type':'TVH','training_level':'TLVL','validation_level':'VLVL','holdout_level':'HLVL'}
        self.assertRaisesRegexp(AimValidationException,
            'keys missing for partition type: GroupCV', self.project_service.validate_cv_method, postdata)

        # Test proper submission:
        postdata = {'pid':'1','cv_method':"GroupCV",'validation_type':'CV','partition_key_cols':["Blah1","Blah2"],'reps':10,'holdout_pct':30}
        expected = {'cv_method':'GroupCV','partition_key_cols':["Blah1","Blah2"],'reps':10,'holdout_pct':30,'ui_validation_type':'CV'}
        self.assertEqual(self.project_service.validate_cv_method(postdata), expected)

        # Test bad value:
        postdata = {'pid':'1','cv_method':"GroupCV",'validation_type':'CV','partition_key_cols':["Blah1","Blah2"],'reps':1000000,'holdout_pct':30}
        self.assertRaisesRegexp(AimValidationException,
            'Invalid number of CV folds \(2 - 999999\)', self.project_service.validate_cv_method, postdata)

    def test_validate_dateCV(self):
        # Test improper submission:
        postdata = {'pid':'1','cv_method':"DateCV",'validation_type':'TVH','datetime_col':'PurchDate'}
        self.assertRaisesRegexp(AimValidationException,
            'keys missing for partition type: DateCV', self.project_service.validate_cv_method, postdata)

        # Test proper submission:
        postdata = {'pid':'1','cv_method':"DateCV",'validation_type':'TVH','datetime_col':'PurchDate','time_validation_pct':15,'time_holdout_pct':30}
        expected = {'datetime_col':'PurchDate','time_validation_pct':15,'time_holdout_pct':30,'cv_method':"DateCV",'ui_validation_type':'TVH'}
        self.assertEqual(self.project_service.validate_cv_method(postdata), expected)

        # Test bad value:
        postdata = {'pid':'1','cv_method':"DateCV",'validation_type':'TVH','datetime_col':'PurchDate','time_validation_pct':10,'time_holdout_pct':'cat'}
        self.assertRaisesRegexp(AimValidationException,
            'Invalid number for holdout percentage \(0-98\)%', self.project_service.validate_cv_method, postdata)

    def test_validate_randomseed(self):
        # Test proper submission:
        postdata = {'pid':'1','randomseed':23}
        expected = {'randomseed':23}
        self.assertEqual(self.project_service.validate_randomseed(postdata), expected)

        # Test bad value:
        postdata = {'pid':'1','randomseed':99999999999}
        self.assertRaisesRegexp(AimValidationException,
            'Invalid value for random seed \(0 - 9999999999\)', self.project_service.validate_randomseed, postdata)

    @pytest.mark.db
    def test_get_project_info_for_user(self):
        target = 'TargetName'
        self.project_service.set_submitted_target(target)

        project = self.project_service.get()

        project = self.project_service.get_project_info_for_user(project, self.uid)
        self.assertIn('target', project)
        self.assertNotIn('submitted_target', project)
        self.assertEqual(project['target']['name'], target)


    @pytest.mark.db
    def test_deactivate_model_with_default_instance(self):
        instance = SHARED_PREDICTION_API_KEYWORD
        instance_id = ObjectId()
        api_instances = [
            {
            '_id': instance,
                    'activation_status': 3,
                    'activated_on' : '1402346958649'
                },
            {
            '_id': instance_id,
            'activation_status': 1,
            'activated_on' : '1402346958649'
            }
        ]

        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={'pid':self.pid, 'api_instances': api_instances, 'api_activated':2})

        self.project_service.deactivate_model(lid, SHARED_PREDICTION_API_KEYWORD)


        model = self.persistent.read(table = LEADERBOARD_TABLE, result = {},
            condition = {'_id': lid})

        api_instances = model.get('api_instances')
        self.assertTrue(api_instances)
        self.assertEqual(model.get('api_activated'), 1)
        self.assertEqual(api_instances[0]['_id'], instance_id)


    @pytest.mark.db
    def test_deactivate_model(self):
        instance_id_1 = ObjectId('5383e637ab6f3c8e17ba622b')
        instance_id_2 = ObjectId('53961ddb0f8ff7c924d0f4b0')
        api_instances = [
            {
            '_id': instance_id_1,
                    'activation_status': 3,
                    'activated_on' : '1402346958649'
                },
            {
            '_id': instance_id_2,
            'activation_status': 1,
            'activated_on' : '1402346958649'
            }
        ]

        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={
            'pid':self.pid,
            'api_instances' : api_instances,
            'api_activated' : 2,
        })

        self.project_service.deactivate_model(lid, instance_id_1)
        self.project_service.deactivate_model(lid, instance_id_2)

        model = self.persistent.read(table = LEADERBOARD_TABLE, result = {},
            condition = {'_id': lid})

        api_instances = model.get('api_instances')
        self.assertFalse(api_instances)
        self.assertFalse(model.get('api_activated'))

    @pytest.mark.db
    def test_deactivate_model_does_not_set_api_activated_with_negative(self):
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={
            'pid':self.pid,
            'api_activated' : 0,
        })

        self.project_service.deactivate_model(lid, ObjectId())

        model = self.persistent.read(table = LEADERBOARD_TABLE, result = {},
            condition = {'_id': lid})

        self.assertEqual(model.get('api_activated'), 0)

    @pytest.mark.db
    def test_activate_model(self):
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={'pid':self.pid})
        instance_id_1 = ObjectId()
        instance_id_2 = ObjectId()

        self.project_service.activate_model(lid, instance_id_1)
        self.project_service.activate_model(lid, instance_id_2)


        model = self.persistent.read(table = LEADERBOARD_TABLE, result = {},
            condition = {'_id': lid})

        self.assertTrue(model.get('api_activated'))

        api_instances = model.get('api_instances')
        self.assertEqual(len(api_instances), 2, api_instances)

        for iid in [instance_id_1, instance_id_2]:
            instance_model_data = next(i for i in api_instances if i['_id'] == iid)
            self.assertTrue(instance_model_data.get('activation_status'))
            self.assertTrue(instance_model_data.get('activated_on'))


    @pytest.mark.db
    def test_activate_model_with_default_instance(self):
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={'pid':self.pid})
        instance = SHARED_PREDICTION_API_KEYWORD
        self.project_service.activate_model(lid, SHARED_PREDICTION_API_KEYWORD)


        model = self.persistent.read(table = LEADERBOARD_TABLE, result = {},
            condition = {'_id': lid})

        api_instances = model.get('api_instances')
        self.assertTrue(api_instances)
        self.assertEqual(model.get('api_activated'), 1)

    @pytest.mark.db
    def test_activate_model_with_default_instance_and_dedicated_instance(self):
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={'pid':self.pid})
        instance = SHARED_PREDICTION_API_KEYWORD
        instance_id = ObjectId()

        self.project_service.activate_model(lid, SHARED_PREDICTION_API_KEYWORD)
        self.project_service.activate_model(lid, instance_id)

        model = self.persistent.read(table = LEADERBOARD_TABLE, result = {},
            condition = {'_id': lid})

        api_instances = model.get('api_instances')
        self.assertTrue(api_instances)
        self.assertEqual(len(api_instances), 2)
        self.assertEqual(model.get('api_activated'), 2)


    @pytest.mark.db
    def test_cannot_activate_same_model(self):
        instance_id = ObjectId()
        instances = [instance_id, instance_id]

        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={'pid':self.pid})

        for i in instances:
            self.project_service.activate_model(lid, i)
            model = self.project_service.get_api_instances_for_model(lid)
            api_instances = model.get('api_instances')
            self.assertTrue(api_instances)
            self.assertEqual(model.get('api_activated'), 1)

    @pytest.mark.db
    def test_get_api_instances_for_model(self):
        api_instances = [
            {
                '_id': ObjectId('5383e637ab6f3c8e17ba622b'),
                'activation_status': 3,
                'activated_on' : '1402346958649'
            },
            {
                '_id': ObjectId('53961ddb0f8ff7c924d0f4b0'),
                'activation_status': 1,
                'activated_on' : '1402346958649'
            }
        ]

        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={'pid':self.pid,
            'api_instances': api_instances})

        model = self.project_service.get_api_instances_for_model(lid)

        self.assertEqual(model.get('api_instances'), api_instances, model)

    def test_activate_deactivate_and_get_list(self):
        lid = self.persistent.create(table=LEADERBOARD_TABLE, values={'pid':self.pid})

        instances = [ObjectId(), ObjectId()]

        counter = 0
        for i in instances:
            counter = counter + 1
            self.project_service.activate_model(lid, i)
            model = self.project_service.get_api_instances_for_model(lid)
            api_instances = model.get('api_instances')
            self.assertTrue(api_instances)
            self.assertEqual(len(api_instances), counter)
            self.assertEqual(model.get('api_activated'), counter)

        for i in instances:
            self.project_service.deactivate_model(lid, i)
            self.project_service.deactivate_model(lid, i)

        model = self.project_service.get_api_instances_for_model(lid)
        self.assertFalse(model.get('api_instances'))
        self.assertEqual(model.get('api_activated'), 0)

    def test_get_models_for_instance(self):
        instance_1 = ObjectId()
        instance_2 = ObjectId()

        api_instances = [
            {
                '_id': instance_1,
                'activation_status': 3,
                'activated_on' : '1402346958649'
            },
            {
                '_id': instance_2,
                'activation_status': 1,
                'activated_on' : '1402346958649'
            }
        ]

        lid_1 = self.persistent.create(table=LEADERBOARD_TABLE, values={
            'pid':self.pid,
            'api_instances': api_instances
        })

        lid_2 = self.persistent.create(table=LEADERBOARD_TABLE, values={
            'pid':self.pid,
            'api_instances': api_instances
        })

        lid_3 = self.persistent.create(table=LEADERBOARD_TABLE, values={
            'pid':self.pid,
            'api_instances': api_instances[:1]
        })

        models = self.project_service.get_models_for_instance(instance_1)
        self.assertTrue(models)
        self.assertEqual(len(models), 3)

        models = self.project_service.get_models_for_instance(instance_2)
        self.assertTrue(models)
        self.assertEqual(len(models), 2)


class TestProjectCaching(unittest.TestCase):

    @pytest.mark.unit
    @patch('common.services.eda.assert_valid_id')
    def test_get_caches_data(self, vmock):
        pid = '5223deadbeefdeadbeef5223'
        uid = '5332deadfedbeafadded5332'
        tempstore_mock = Mock()
        pers_mock = Mock()
        p = ProjectService(pid, uid, tempstore=tempstore_mock,
                           persistent=pers_mock)

        pers_mock.read.return_value = [{'pretend_field': 'pretend_value'}]
        #Act
        p.get()

        #Assert
        self.assertIn('pretend_field', p.data)
        self.assertEqual(p.data['pretend_field'], 'pretend_value')

class TestProjectModelFrontEnd(unittest.TestCase):

    @pytest.mark.unit
    def test_get_label_for_RFC_is_extra_trees(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=1;c=0'], 'P']}
        self.assertEqual(task_descriptions['RFC']['label'], 'DYNAMIC')
        bp_info = ProjectService().get_blueprint_data_for_bp(blueprint)
        self.assertEqual(task_descriptions['RFC']['label'], 'DYNAMIC')
        front_taskmap = bp_info['taskMap']
        labels = [front_taskmap[key]['label'] for key in front_taskmap.keys()]
        self.assertIn('ExtraTrees Classifier (Gini)', labels)
        self.assertEqual(task_descriptions['RFC']['label'], 'DYNAMIC')

    @pytest.mark.unit
    def test_get_label_for_RFC_entropy_is_random_forest(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1'], 'P']}
        bp_info = ProjectService().get_blueprint_data_for_bp(blueprint)
        front_taskmap = bp_info['taskMap']
        labels = [front_taskmap[key]['label'] for key in front_taskmap.keys()]
        self.assertIn('RandomForest Classifier (Entropy)', labels)
        self.assertEqual(task_descriptions['RFC']['label'], 'DYNAMIC')

    @pytest.mark.unit
    def test_get_label_works_with_spaces_in_metric(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC tm=Gamma Deviance;e=0;c=1'], 'P']}
        bp_info = ProjectService().get_blueprint_data_for_bp(blueprint)
        front_taskmap = bp_info['taskMap']
        labels = [front_taskmap[key]['label'] for key in front_taskmap.keys()]
        self.assertIn('RandomForest Classifier (Entropy)', labels)
        self.assertEqual(task_descriptions['RFC']['label'], 'DYNAMIC')

    @pytest.mark.skip
    def test_get_label_for_LR1_with_penalty(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['LR1 p=0'], 'P']}

        bp_info = ProjectService().get_blueprint_data_for_bp(blueprint)
        rfc_label = bp_info['taskMap']['LR1']['label']
        self.assertEqual(rfc_label,'Regularized Logistic Regression (L1)')

    @pytest.mark.skip
    def test_get_url_for_LR1(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['LR1 p=0'], 'P']}

        bp_info = ProjectService().get_blueprint_data_for_bp(blueprint)

        self.assertIn('url', bp_info['taskMap']['LR1'] )

    @pytest.mark.unit
    def test_get_bp_diagram_falls_back_to_blackbox(self):
        '''If any error occurs when trying to load a stored blueprint diagram,
        default to a very blackboxy representation
        '''
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['NUM'], ['GS'], 'T'],
                     '3': [['CAT'], ['DM2'], 'T'],
                     '4': [['1', '2', '3'], ['RFC'], 'P']}

        pservice = ProjectService()

        with patch.object(pservice, 'get_blueprint_menu') as fake_menu:
            fake_menu.return_value = {'will_cause': 'problems'}
            leaderboard_item = {'blueprint': blueprint,
                                'blueprint_id': 'doesnot matter for this test'}

            dia = pservice.get_blueprint_diagram_from_blueprint_id(
                    leaderboard_item['blueprint_id'])

            import pprint
            pprint.pprint(dia)

            self.assertNotIn('DM2', dia['taskMap'].keys())
            self.assertIn('PREP', dia['taskMap'].keys())
            self.assertEqual(dia['taskMap']['DR_F']['label'], 'DataRobot Custom Model')

    @pytest.mark.integration
    def test_blender_diagram_manual_mode(self):
        '''On manual mode, `bp` numbers are not stored with models inside the
        menu, so the way we were looking up models before was breaking the
        leaderboard on manual mode

        '''
        #SETUP
        blueprints = [{'1': [['NUM'], ['NI'], 'T'],
                       '2': [['1'], ['RFC'], 'P']},
                      {'1': [['NUM'], ['NI'], 'T'],
                       '2': [['1'], ['SVMC'], 'P']},
                      {'1': [['NUM'], ['NI'], 'T'],
                       '2': [['1'], ['LR1'], 'P']}]
        blueprint_ids = ['fde8e8577e3103d89fabec0261786a55',
                         'fbef14465c94afb274880ff798a7d18b',
                         'b0b7a82828713798a4cd9e998433ff24']
        bps = ['3', '5', '7']

        leaderboard = [{'blueprint':blue,
                        'bp':bp,
                        'blueprint_id': bp_id,
                        'samplepct': x} for x in [32, 48, 64]
                        for blue, bp, bp_id in zip(blueprints,
                                                   bps,
                                                   blueprint_ids)]

        blueprint = {
                '1': [['b67aee14ac38a5bd0a8498abe3cb278e'], ['MEDBL'], 'P']}
        leaderboard_item = {
            'bp': '3+5+7',
            'blender': {
                'inputs': [{'blender': {},
                            'blueprint': blueprints[0],
                            'dataset_id': '5223beefbeefdeadbeef5223',
                            'samplepct': 64},
                           {'blender': {},
                            'blueprint': blueprints[1],
                            'dataset_id': '5223beefbeefdeadbeef5223',
                            'samplepct': 64},
                           {'blender': {},
                            'blueprint': blueprints[2],
                            'dataset_id': '5223beefbeefdeadbeef5223',
                            'samplepct': 64}]}}
        pservice = ProjectService()
        with patch.multiple(pservice, get_blueprint_menu=DEFAULT,
                            raw_leaderboard_read=DEFAULT) as fakes:
            fakes['get_blueprint_menu'].return_value = {
                'fde8e8577e3103d89fabec0261786a55': {'bp': 1},
                'fbef14465c94afb274880ff798a7d18b': {'bp': 2},
                'b0b7a82828713798a4cd9e998433ff24': {'bp': 3},
            }
            fakes['raw_leaderboard_read'].return_value = leaderboard
            dia = pservice.flippered_diagram_selection(blueprint,
                                                       leaderboard_item)

        labels = [dia['taskMap'][key]['label']
                  for key in dia['taskMap'].keys()]
        self.assertIn('ExtraTrees Classifier (Gini)', labels)
        self.assertIn('Support Vector Classifier (Radial Kernel)', labels)
        self.assertIn('Regularized Logistic Regression (L2)', labels)
        self.assertIn('Median Blender', labels)

    @pytest.mark.integration
    def test_blender_diagram_issue_1993(self):
        '''Dynamic names gotchas, I think

        '''
        #SETUP
        blueprints = [
            {u'1': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'RFR e=0'], u'P']},
            {u'1': [[u'NUM'], [u'NI'], u'T'],
             u'2': [[u'1'], [u'BTRANSF logy;d=1'], u'T'],
             u'3': [[u'2'], [u'ST'], u'T'],
             u'4': [[u'3'], [u'RIDGE logy;cs_l=0.69314718056;t_m=RMSE'], u'S'],
             u'5': [[u'4'], [u'CALIB f=Gamma;e=GLM;p=2'], u'P']},
            {u'1': [[u'NUM'], [u'NI'], u'T'],
             u'2': [[u'1'], [u'BTRANSF logy;dist=3;d=2'], u'T'],
             u'3': [[u'2'], [u'ST'], u'T'],
             u'4': [[u'3'], [u'LASSO2 logy;'], u'T'],
             u'5': [[u'4'], [u'GLMR d=2'], u'P']}]

        blueprint_ids = [
            'b43f29f571aad1ae814475dd55b9dc2f',
            'c114bfb0b33aaab65a8e32f536f7f03b',
            'd1ae18ac635d8d404fb4953a1b106378']

        bps = ['4', '18', '21']

        leaderboard = [{'blueprint':blue,
                        'bp':bp,
                        'blueprint_id': bp_id,
                        'samplepct': x} for x in [32, 48, 64]
                        for blue, bp, bp_id in zip(blueprints,
                                                   bps,
                                                   blueprint_ids)]

        blueprint = {
                '1': [['b67aee14ac38a5bd0a8498abe3cb278e'], ['MEDBL'], 'P']}
        leaderboard_item = {
            'bp': '4+18+21',
            'blender': {
                'inputs': [{'blender': {},
                            'blueprint': blueprints[0],
                            'dataset_id': '5223beefbeefdeadbeef5223',
                            'samplepct': 64},
                           {'blender': {},
                            'blueprint': blueprints[1],
                            'dataset_id': '5223beefbeefdeadbeef5223',
                            'samplepct': 64},
                           {'blender': {},
                            'blueprint': blueprints[2],
                            'dataset_id': '5223beefbeefdeadbeef5223',
                            'samplepct': 64}]}}
        pservice = ProjectService()
        with patch.multiple(pservice, get_blueprint_menu=DEFAULT,
                            raw_leaderboard_read=DEFAULT) as fakes:
            fakes['get_blueprint_menu'].return_value = {
                blueprint_ids[0]: {'bp': bps[0]},
                blueprint_ids[1]: {'bp': bps[1]},
                blueprint_ids[2]: {'bp': bps[2]},
            }
            fakes['raw_leaderboard_read'].return_value = leaderboard
            dia = pservice.flippered_diagram_selection(blueprint,
                                                       leaderboard_item)
        self.assertIn('MEDBL', dia['taskMap'])
        labels = [dia['taskMap'][key]['label']
                  for key in dia['taskMap'].keys()]
        self.assertIn('RandomForest Regressor', labels)
        self.assertIn('Generalized Linear Model (Gamma Distribution)', labels)
        self.assertIn('Ridge Regression', labels)

    @pytest.mark.integration
    def test_blender_diagram_issue_2012(self):
        '''More dynamic names gotchas, I think

        '''
        #SETUP
        blueprints = [
                {u'1': [[u'NUM'], [u'NI'], u'T'],
                 u'2': [[u'1'],
                        [u'RFR logy;cs_l=0.69314718056;e=0;t_a=2;t_n=1;'
                          't_f=0.15;ls=[5, 10, 20];mf=[0.2, 0.3, 0.4, 0.5];'
                          't_m=RMSE'],
                        u'S'],
                  u'3': [[u'2'], [u'CALIB f=Gamma;e=GLM;p=2'], u'P']},
                {u'1': [[u'NUM'], [u'NI'], u'T'],
                 u'2': [[u'1'],
                        [u'RFR logy;cs_l=0.69314718056;e=1;t_a=2;t_n=1;'
                          't_f=0.15;ls=[5, 10, 20];mf=[0.2, 0.3, 0.4, 0.5];'
                          't_m=RMSE'],
                         u'S'],
                 u'3': [[u'2'], [u'CALIB f=Gamma;e=GLM;p=2'], u'P']}]

        blueprint_ids = [blueprint_id(bp) for bp in blueprints]

        bps = ['13', '14']

        leaderboard = [{'blueprint':blue,
                        'bp':bp,
                        'blueprint_id': bp_id,
                        'samplepct': x} for x in [32, 48, 64]
                        for blue, bp, bp_id in zip(blueprints,
                                                   bps,
                                                   blueprint_ids)]

        blueprint = {
                '1': [['b67aee14ac38a5bd0a8498abe3cb278e'], ['MEDBL'], 'P']}
        leaderboard_item = {
            'bp': '+'.join(bps),
            'blender': {
                'inputs': [{'blender': {},
                            'blueprint': bp,
                            'dataset_id': '5223beefbeefdeadbeef5223',
                            'samplepct': 64} for bp in blueprints]
                        }}
        pservice = ProjectService()
        with patch.multiple(pservice, get_blueprint_menu=DEFAULT,
                            raw_leaderboard_read=DEFAULT) as fakes:
            fakes['get_blueprint_menu'].return_value = {
                blueprint_ids[i]: {'bp': bps[i]}
                for i in range(len(blueprints))
            }
            fakes['raw_leaderboard_read'].return_value = leaderboard
            dia = pservice.flippered_diagram_selection(blueprint,
                                                       leaderboard_item)
        labels = [dia['taskMap'][key]['label']
                  for key in dia['taskMap'].keys()]
        self.assertIn('RandomForest Regressor', labels)
        self.assertIn('ExtraTrees Regressor', labels)
        self.assertNotIn('Calibrate predictions', labels)

    def test_load_diagram_will_retroactively_patch_issue_2024(self):
        bad_dia = u'{"taskMap": {"START": {"converter_inputs": "", "description": "", "url": "", "target_type": "", "label": "Data", "version": "", "arguments": "", "model_family": "", "sparse_input": "", "icon": ""}, "GLMB": {"label": {"converter_inputs": null, "description": "Generalized Linear Model Regression (Bernoulli Distribution)", "url": null, "target_type": "b", "label": "Generalized Linear Model (Bernoulli Distribution)", "version": "0.1", "arguments": {"p": {"default": "1.5", "values": [0, 2], "type": "floatgrid", "name": "p"}, "tl": {"default": "1", "values": [false, true], "type": "select", "name": "tweedie_log"}, "d": {"default": "2", "values": ["Gaussian", "Poisson", "Bernoulli", "Gamma", "Tweedie"], "type": "select", "name": "distribution"}}, "model_family": "GLM", "sparse_input": false, "icon": 0}}, "PREP": {"label": "Generalized Linear Model Preprocessing v7"}, "END": {"converter_inputs": "", "description": "", "url": "", "target_type": "", "label": "Prediction", "version": "", "arguments": "", "model_family": "", "sparse_input": "", "icon": ""}}, "tasks": ["START"], "type": "start", "id": "0", "children": [{"inputs": ["0"], "tasks": ["PREP"], "id": "1", "output": "P", "type": "input", "children": [{"inputs": ["1"], "tasks": ["GLMB"], "id": "2", "output": "P", "type": "task", "children": [{"inputs": ["2"], "tasks": ["END"], "type": "end", "id": "3", "output": " "}]}]}]}'
        dia = project_module.load_diagram(bad_dia)

        self.assertEqual(dia['taskMap']['GLMB']['label'],
                         'Generalized Linear Model (Bernoulli Distribution)')

    @patch('MMApp.entities.project.FLIPPERS', autospec = True)
    def test_should_censor_insights(self, MockFlippers):
        MockFlippers.graybox_enabled.return_value = True

        project_service = ProjectService()
        lb = {'insights': 'something'}
        result = project_service.should_censor_insights(lb)
        self.assertFalse(result)


        lb = {'reference_model': 'x'}
        result = project_service.should_censor_insights(lb)
        self.assertFalse(result)

        lb = {'insights' : 'NA', 'reference_model': False, 'other-keys': 'x' }
        result = project_service.should_censor_insights(lb)
        self.assertTrue(result)


    def test_censored_insights(self):
        """Test if censoring insights works as expected. """
        project_service = ProjectService()
        with patch.object(project_service, 'get_original_featnames') as mock_orig_feat:
            mock_orig_feat.return_value = ['foo', 'bar']

            # pass through unmodified if only raw features
            leaderboard_item = {'pid': 'foobar',
                                'extras': {u'(0, -1)': {'coefficients': [['foo', 1.0],
                                                                        ['bar', 1.0]]}}}

            res = project_service.censored_insights(leaderboard_item)
            self.assertDictEqual(leaderboard_item, res)

            # censor non-raw features
            leaderboard_item_w_non_orig = {'pid': 'foobar',
                                           'extras': {u'(0, -1)': {
                                               'coefficients': [['foo', 1.0],
                                                                ['bar', 1.0],
                                                                ['blablub', 1.0]]}}}

            res = project_service.censored_insights(leaderboard_item_w_non_orig)

            expected_res = {'pid': 'foobar', 'extras': {u'(0, -1)': {
                                               'coefficients': [['foo', 1.0],
                                                                ['bar', 1.0],
                                                                ['Derived Feature 1', 1.0]]}}}
            self.assertDictEqual(expected_res, res)

            # ignore hotspots
            leaderboard_item_w_hotspots = {'pid': 'foobar',
                                           'extras': {u'(0, -1)': {
                                               'coefficients': [['foo', 1.0],
                                                                ['bar', 1.0]],
                                               'hotspots': [{'importance': 0.1, 'support': 0.1}]}}}
            res = project_service.censored_insights(leaderboard_item_w_hotspots)
            self.assertDictEqual(leaderboard_item_w_hotspots, res)

    @pytest.mark.db
    def test_get_insights(self):
        persistent = database.new('persistent')
        uid = ObjectId()
        pid = persistent.create(table='project', values = {'uid': str(uid)})

        project_service = ProjectService (pid,uid)

        leaderboard = [
            {'insights' : 'hotspots'}, # bueno
            {'insights' : 'NA',
                'extras' : {
                    '(0, -1)' : {
                        'coefficients' : [
                            [
                                'VehOdo',
                                1
                            ],
                        ]
                    }
                }
            }, # no bueno
            {'insights' : 'NA',
                'extras' : {
                    '(0, -1)' : {
                        'importance' : [
                            [
                                'VehOdo',
                                1
                            ],
                        ]
                    }
                }
            }, # bueno
            {'insights' : 'NA',
                'extras' : {
                    '(-1, -1)' : {
                        'importance' : [
                            [
                                'VehOdo',
                                1
                            ],
                        ]
                    }
                }
            }, # bueno
        ]

        for item in leaderboard:
            item['pid'] = pid
            item['reference_model'] = False
            persistent.create(table=LEADERBOARD_TABLE, values = item)

        persistent.create(table='metadata', values = {'pid': pid})

        insights = project_service.get_insights()

        self.assertTrue(insights)
        self.assertEqual(len(insights), 3)


    @pytest.mark.unit
    def test_get_leaderboard_blueprints(self):

        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                       '2': [['1'], ['RFC'], 'P']}
        leaderboard = [
            {'_id': 1, 'blueprint' : blueprint},
            {'_id': 2, 'blueprint' : blueprint},
            {'_id': 3, 'blueprint' : blueprint},
        ]

        project_service = ProjectService(ObjectId(), ObjectId())
        with patch.multiple(project_service, persistent = DEFAULT, flippered_diagram_selection = DEFAULT) as mocks:
            mocks['persistent'].read.return_value = leaderboard

            result = project_service.get_leaderboard_blueprints()

            self.assertTrue(result)
            self.assertEqual(len(result), 3)


class TestValidMetricsFiltering(unittest.TestCase):

    def test_filter_with_errored_job_returns_job(self):
        # Issue #3035
        leaderboard_item = {
            'blueprint': 'blah',
            'blueprint_id': 'blah',
        }   # Main issue - `test` key is missing

        valid_metrics = ['Gini']

        filtered = project_module.with_only_valid_metrics(leaderboard_item,
                                                          valid_metrics)

        self.assertEqual(filtered, leaderboard_item)

    def test_filters_unknown_metrics(self):
        leaderboard_item = {
            'blueprint': 'blah',
            'blueprint_id': 'blah',
            'test': {'metrics': ['Gini', 'InappropriateMetric'],
                     'Gini': [0.5],
                     'InappropriateMetric': [0.6]
                    }
        }

        valid_metrics = ['Gini']

        filtered = project_module.with_only_valid_metrics(leaderboard_item,
                                                          valid_metrics)

        self.assertNotIn('InappropriateMetric', filtered['test'])


if __name__ == '__main__':
    unittest.main()

