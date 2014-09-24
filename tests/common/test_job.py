import copy
import unittest
import itertools
import json
import pytest
from mock import Mock, patch
from bson.objectid import ObjectId

from common.entities.job import DataRobotJob, BlenderRequest, find_model_in_queue
from common.entities.job import ModelRequest
import common.entities.job as jobmod
from common.entities.blueprint import Icons

class TestFindModelInQueue(unittest.TestCase):

  def setUp(self):
    self.queue_snapshot = [
      {'status': 'settings', 'pause': True, 'qid': -1, 'workers': 2, 'done': False, 'mode': 0, 'dataset_id': u'53f26094cb0936483840c604'},
      {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'3': [[u'1', u'2'], [u'ST'], u'T'], u'2': [[u'CAT'], [u'DM2 cm=1000'], u'T'], u'4': [[u'3'], [u'LR1 p=0'], u'P']}, u'lid': u'53f260a1cb0936484d40c604', u'samplepct': 64, u'uid': u'53ee82263092438fee3b2c87', u'blueprint_id': u'ae3e2379a7f851d827ae3759234787b1', u'total_size': 158.0, u'qid': 3, u'icons': [1], u'pid': u'53f26091cb093632c00bafe6', u'max_reps': 1, 'status': 'queue', u'bp': 3, u'model_type': u'Regularized Logistic Regression (L1)', u'dataset_id': u'53f26094cb0936483840c604', u'new_lid': True, u'max_folds': 0, u'reference_model': True, u'features': [u'Missing Values Imputed', u'One-Hot Encoding', u'Standardize', u'Regularized Logistic Regression (L1)']},
      {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'3': [[u'1', u'2'], [u'ST'], u'T'], u'2': [[u'CAT'], [u'DM2 cm=1000'], u'T'], u'4': [[u'3'], [u'LR1 p=1'], u'P']}, u'lid': u'53f260a1cb0936484d40c605', u'samplepct': 64, u'uid': u'53ee82263092438fee3b2c87', u'blueprint_id': u'e839218c0eff338b14a708e81cdec46e', u'total_size': 158.0, u'qid': 4, u'icons': [1], u'pid': u'53f26091cb093632c00bafe6', u'max_reps': 1, 'status': 'queue', u'bp': 4, u'model_type': u'Regularized Logistic Regression (L2)', u'dataset_id': u'53f26094cb0936483840c604', u'new_lid': True, u'max_folds': 0, u'reference_model': True, u'features': [u'Missing Values Imputed', u'One-Hot Encoding', u'Standardize', u'Regularized Logistic Regression (L2)']}
    ]

    self.prediction_request = {
      'scoring_dataset_id': u'53f33c31cb09366187bf8852',
      'blueprint': {u'1': [[u'CAT'], [u'DM2 sc=10;cm=10000'], u'T'], u'3': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'BNBC '], u'T'], u'5': [[u'2', u'4'], [u'CNBC'], u'S'], u'4': [[u'3'], [u'GNBC'], u'T'], u'6': [[u'5'], [u'CALIB f=binomial;e=GLM;p=2;l=logit'], u'P']},
      'samplepct': 64,
      'uid': '53ee82263092438fee3b2c87',
      'blueprint_id': u'c07a3342d5fab1d2e913a2d460a70ecf',
      'predict': 1,
      'dataset_id': u'53f26094cb0936483840c604',
      'pid': '53f26091cb093632c00bafe6',
      'partitions': []
    }

  def test_new_model_request(self):
    model_1 = {
      'pid': ObjectId('53f26091cb093632c00bafe6'),
      'blueprint_id': '123',
      'dataset_id': '53f26091cb093632c00bafe6',
      'samplepct': '10'
    }

    found = find_model_in_queue(model_1, self.queue_snapshot)
    self.assertFalse(found)

  def test_duplicate_model_request(self):
    model_1 = self.queue_snapshot[1].copy()
    found = find_model_in_queue(model_1, self.queue_snapshot)
    self.assertTrue(found)

  def test_prediction_request(self):
    found = find_model_in_queue(self.prediction_request, self.queue_snapshot)
    self.assertFalse(found)

  def test_second_prediction_request(self):
    existing_prediction_request = self.prediction_request.copy()
    existing_prediction_request['max_reps'] = 1
    existing_prediction_request['scoring_dataset_id'] = '53b093f33c66187bf885231c'
    self.queue_snapshot.append(existing_prediction_request)

    found = find_model_in_queue(self.prediction_request, self.queue_snapshot)
    self.assertFalse(found)

class TestJobClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.blueprint = {"1": [["NUM"], ["NI"], "T"],
                         "2": [["1"], ["LR1"], "P"]}
        cls.partitions = [list(i) for i in itertools.product(range(5), [-1])]
        cls.as_dict = {'blueprint': cls.blueprint,
                       'partitions': cls.partitions,
                       'dataset_id': 'A dataset_id',
                       'blueprint_id': 'A blueprint_id',
                       'pid': 'A pid',
                       'uid': 'A uid',
                       'lid': 'new'}
        cls.as_json = json.dumps(cls.as_dict)
        cls.full_job = DataRobotJob(cls.as_dict)
        #cls.full_job.set_samplesize(1000)

    @pytest.mark.unit
    def test_can_initialize_incomplete_from_dict(self):
        d = copy.deepcopy(self.as_dict)
        d['thisisnotused'] = 1
        j = DataRobotJob(d)

        self.assertEqual(j.get('blueprint'), self.blueprint)
        self.assertEqual(len(j.get('partitions')), 5)

        self.assertIsNone(j.get('thisisnotused'))

    @pytest.mark.unit
    def test_can_initialize_from_json(self):
        #Passes if no error
        j = DataRobotJob.from_json(self.as_json)

    @pytest.mark.unit
    def test_to_json_reversible(self):
        j = self.full_job
        k = DataRobotJob.from_json(j.to_json())
        self.assertEqual(k.get('blueprint'), j.get('blueprint'))
        self.assertEqual(k.get('dataset_id'), j.get('dataset_id'))
        self.assertEqual(k.get('partitions'), j.get('partitions'))

    @pytest.mark.unit
    def test_samplepct_is_integer(self):
        d = copy.deepcopy(self.as_dict)
        d['samplepct'] = 60
        d['total_size'] = 1294.0

        k = DataRobotJob(d)
        self.assertIsInstance(k.get('samplepct'), int)
        self.assertEqual(k.get('samplepct'), 60)

class TestModelRequeset(unittest.TestCase):

    def setUp(self):
        self.pid = ObjectId('5223deadbeefdeadbeef1234')
        self.uid = ObjectId('5233deadbeefdeadbeef0000')
        self.partition = {u'total_size': 200, u'folds': 5,
                          u'cv_method':u'RandomCV', u'reps': 5,
                          u'holdout_pct': 20}

    def test_validate_request_inserts_python_icon_for_python_user_items(self):
        '''Jobs which are submitted from the user tasks repo to app.py do not
        come with icons.  But it is pretty clear the correct icon to submit,
        based upon the selectedCode attribute - so we just need to take the
        opportunity during validation to determine the correct icon to submit
        '''
        request = {
            'max_reps':1,
            'dataset_id':'53a8702c8bd88f363af98732',
            'task_id':'53a870648bd88f35bc0d0ac6',
            'samplepct':32,
            'selectedCode':{
                'created':'2014-06-23 14:22:28.286718',
                'modelpredict':'',
                'modelsource':"import pandas as pd\nimport numpy as np\n\nclass CustomModel(object):\n\n def fit(self, X, Y):\n '''This is where you would fit your model to the data\n\n Parameters\n ----------\n X : pandas.DataFrame\n Contains your data - you can think of it as simply loaded from\n pandas.read_csv, but any transformed or derived features you\n have included come along\n Y : pandas.Series\n Has as many elements as X has rows - these are the predictions\n that go along with the data in X\n\n Returns\n -------\n self : CustomModel\n We utilize operator chaining and need to be able to run\n ``self.fit().predict()`` or similar.\n '''\n return self\n\n def predict(self, X):\n '''This is the prediction method that you would call\n\n The output is a numpy ndarray, having a single column and as many\n rows as X\n\n Parameters\n ----------\n X : pandas.DataFrame\n The data on which to make a prediction using your newly fit model\n \n Returns\n -------\n Y : numpy.ndarray\n With a single column, and the same number of rows as ``X``\n '''\n return np.ones((len(X), 1))",
                'modeltype':'Python',
                'classname':'CustomModel',
                'modelfit':'',
                'version_id':'53a870648bd88f35bc0d0ac7'},
            'model_type':'IPython Model 1',
            'features':[]}

        mr = ModelRequest(request, self.pid, self.uid, self.partition)
        validated = mr.validate_request(request)
        self.assertIn('icons', validated.keys())
        self.assertEqual(validated['icons'][0], Icons.PYTHON_ICON)

    def test_validate_request_inserts_R_icon_for_R_user_items(self):
        '''Jobs which are submitted from the user tasks repo to app.py do not
        come with icons.  But it is pretty clear the correct icon to submit,
        based upon the selectedCode attribute - so we just need to take the
        opportunity during validation to determine the correct icon to submit
        '''
        request = {"max_reps":1,
                   "dataset_id":"53a877f28bd88f3b99c527da",
                   "task_id":"53a87c208bd88f3b0ffa7d90",
                   "samplepct":32,
                   "selectedCode":{"created":"2014-06-23 15:12:32.284860",
                       "modelpredict":"function(model,data) {\n    predictions = model(data);\n    return(predictions);\n}",
                       "modelsource":"",
                       "modeltype":"R",
                       "classname":"",
                       "modelfit":"function(response,data) {\n    model = function(data) {\n        prediction = rep(0.5,dim(data)[1]);\n        return(prediction);\n    };\n    return(model);\n}",
                       "version_id":"53a87c208bd88f3b0ffa7d91"},
                   "model_type":"RStudio model 1",
                   "features":[]}

        mr = ModelRequest(request, self.pid, self.uid, self.partition)
        validated = mr.validate_request(request)
        self.assertIn('icons', validated.keys())
        self.assertEqual(validated['icons'][0], Icons.R_ICON)

class TestBlenderRequest(unittest.TestCase):

    def test_convert_to_stack_changes_correct_task(self):
        blueprint = {
            u'1': [[u'CAT'], [u'DM2 sc=10;cm=32;dc=1'], u'T'],
            u'10': [[u'9'], [u'GLMB'], u'P'],
            u'2': [[u'1'], [u'ST'], u'T'],
            u'3': [[u'CAT'], [u'CRED1 cmin=33'], u'T'],
            u'4': [[u'3'], [u'LINK l=1'], u'T'],
            u'5': [[u'4'], [u'ST'], u'T'],
            u'6': [[u'NUM'], [u'NI'], u'T'],
            u'7': [[u'6'], [u'BTRANSF dist=2;d=2'], u'T'],
            u'8': [[u'7'], [u'ST'], u'T'],
            u'9': [[u'2', u'5', u'8'], [u'LR1 p=0'], u'T']
        }
        converted = jobmod.change_to_stack(blueprint)
        self.assertEqual(converted['10'][2], 'S')
        self.assertEqual(converted['9'][2], 'T')

    def test_convert_to_stack_basic_case(self):
        blueprint = {
            u'1': [[u'CAT'], [u'DM2 sc=10;cm=32;dc=1'], u'T'],
            u'2': [[u'1'], [u'ST'], u'T'],
            u'3': [[u'2'], [u'RFC'], u'P'],
        }
        converted = jobmod.change_to_stack(blueprint)
        self.assertEqual(converted['3'][2], 'S')

    def test_find_blender_in_queue(self):
        pid = str(ObjectId())
        sig1 = {'pid':pid, 'blueprint_id':'asdf', 'samplepct':50, 'dataset_id':'1234','max_reps':1}
        sig2 = {'pid':pid, 'blueprint_id':'asdf1', 'samplepct':50, 'dataset_id':'1234','max_reps':1}
        sig3 = {'pid':pid, 'blueprint_id':'asdf', 'samplepct':40, 'dataset_id':'1234','max_reps':1}

        out = find_model_in_queue(sig1,[sig2, sig3])
        self.assertFalse(out)

        out = find_model_in_queue(sig2,[sig1,sig2,sig3])
        self.assertTrue(out)

        with patch.object(BlenderRequest, 'validate_request') as mock1:
            mock1.return_value = sig1
            blender = BlenderRequest({},pid,3,Mock())
            out = blender.find_blender_in_queue([])
            self.assertFalse(out)
            out = blender.find_blender_in_queue([sig2,sig3])
            self.assertFalse(out)
            out = blender.find_blender_in_queue([sig1,sig2])
            self.assertTrue(out)


class TestBlenderRequestMethods(unittest.TestCase):

    REQUEST_FIXTURE = {
        "samplepct": 64, "features": [], "icons": [0],
        "model_type": "AVG Blender",
        "blender_request": {"items": ["53810f2a8bd88f63ab3cd34f",
                                      "53810f2a8bd88f63ab3cd35a"],
                            "args": "logitx",
                            "method": "AVG",
                            "family": "binomial"},
        "max_folds": 0,
        "partitions": [[0, -1], [1, -1], [2, -1], [3, -1], [4, -1]],
        "blender": {}}
    PID = ObjectId('53810f048bd88f37914fd724')
    UID = ObjectId('53783ead8bd88f5ae961363b')

    def test_this_fixture_can_still_instantiate_a_blender_request(self):
        fixture = copy.deepcopy(self.REQUEST_FIXTURE)
        blender_request = BlenderRequest(fixture, self.PID, self.UID, {'reps':5, 'holdout_pct':20} )

        # This is a lame assertion, but mostly we care that the validation
        # code is still happy
        self.assertEqual(blender_request['pid'], self.PID)

    def test_reorders_blueprints_as_needed(self):
        '''The blueprint diagram for blenders needs to know which blueprints
        are associated with which bps - but for presentation's sake we reorder
        the bps when creating the bp string (which is something like 4+7+13).
        In order to most easily produce the blueprint diagram for blenders,
        we need to order the blueprints to match the ordering of the bps.
        Before we were just re-ordering the bps
        '''
        fixture = copy.deepcopy(self.REQUEST_FIXTURE)
        blender_request = BlenderRequest(fixture, self.PID, self.UID, {'reps':5, 'holdout_pct': 20})

        item0 = {'blueprint': {'1': [['NUM'], ['NI'], 'T'],
                               '3': [['1', '2'], ['DTC'], 'P'],
                               '2': [['CAT'], ['ORDCAT'], 'T']},
                 'blender': {},
                 'bp': 24,
                 'test': {'LogLoss': [0.01],
                          'metrics': ['LogLoss']},
                 'dataset_id': '53810f068bd88f62703cd350',
                 'samplepct': 64}
        item1 = {'blueprint': {'1': [['NUM'], ['NI'], 'T'],
                               '3': [['1', '2'], ['RFC e=0'], 'P'],
                               '2': [['CAT'], ['ORDCAT'], 'T']},
                 'blender': {},
                 'bp': 21,
                 'test': {'LogLoss': [0.02],
                          'metrics': ['LogLoss']},
                 'dataset_id': '53810f068bd88f62703cd350',
                 'samplepct': 64}

        project_service = Mock()
        project_service.get_leaderboard_item.side_effect = [item0, item1]

        # Act
        blender_request.get_inputs_from_leaderboard(project_service)

        # Assert that the bp and blueprints match up in order
        self.assertEqual(blender_request['bp'], '21+24')
        blender_inputs = blender_request['blender']['inputs']

        self.assertEqual(blender_inputs[0]['blueprint'],
                         jobmod.change_to_stack(item1['blueprint']))
        self.assertEqual(blender_inputs[1]['blueprint'],
                         jobmod.change_to_stack(item0['blueprint']))


if __name__== '__main__':
    unittest.main()
