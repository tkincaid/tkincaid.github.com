import pytest
import unittest
from bson import ObjectId
import json
from mock import patch

from predictionapi.entities.prediction import PredictionService
from predictionapi.entities.prediction import PredictionServiceError
from predictionapi.entities.prediction import PredictionServiceUserError
from predictionapi.prediction_io import PredictionResponse, Prediction


LB_RECORD = {
    "_id": ObjectId("5214d012637aba171e0bbb7a"),
    "pid" : "52dc0ba5637aba2829195fcd",
    "blend": 0,
    "blueprint": {"1": [['ALL'], ['RC'], 'P']},
    "bp": 2,
    "dsid": "5214d011637aba17000bbb7b",

    "features": ["ALL"],
    "hash": "8315b571e7281d5eb801d2eeff3637575d6d3765",
    "icons": [0],
    "max_folds": 0,
    "max_reps": 1,
    "model_type": "Dummy",
    "part_size": [["1", 320, 80]],
    "parts": [["1", "0.06", "3"]],
    "parts_label": [
        "partition",
        "thresh",
        "NonZeroCoefficients"
    ],
    "qid": 2,
    "s": 0,
    "samplepct": 32,
    "task_cnt": 5,
    "total_size": 400,
    "uid": "5214d010637aba16eb0bbb7a",
    "vertex_cnt": 2,
    "wsid": "5214d011637aba17000bbb7a",
}


class PredictionServiceTest(unittest.TestCase):
    """Unit tests for the prediction service.

    Mocks the delegated MMClient and tempstore.
    """


    @pytest.mark.unit
    @patch('predictionapi.entities.prediction.Sentinel')
    @patch('predictionapi.entities.prediction.ProjectService', autospec = True)
    def test_smoke_clf(self, MockProjectService, MockSentinel):
        """Smoke test for prediction service for binary classification. """
        svc = PredictionService(ObjectId(), ObjectId())
        result_id = str(ObjectId())
        res = json.dumps({'status': '', 'success': True, 'result_id' : result_id})
        predictions = {
            'predicted-0': [0.4, 0.8],
            'row_index': [0, 1],
            'actual': [0.0, 0.0],
            'newdata': 'YES',
            'task' : 'Binary'
        }

        project_service = MockProjectService.return_value
        project_service.get_predictions.return_value = predictions

        data = {'mimetype': 'text/plain', 'stream': 'Foo,Bar\n0,1\n1,0\n'}
        with patch.object(svc, 'tempstore') as mock_db:
            with patch.object(svc, 'client') as mock_client:
                mock_client.predict_transient.return_value = None
                mock_db.read.return_value = ('result_id', res)
                MockSentinel.return_value.master_for.return_value.blpop.return_value = ('result-key', res)
                resp = svc.predict(ObjectId(), '52ab7af8b4912910805f6f98', data)
                #self.assertTrue(mock_db.read.called)
                self.assertTrue(mock_client.predict_transient.called)
                self.assertEqual(type(resp), PredictionResponse)
                self.assertEqual(type(resp.predictions[0]), Prediction)
                self.assertEqual(resp.task, 'Binary')
                self.assertEqual(resp.predictions[0].row_id, 0)
                self.assertEqual(resp.predictions[1].row_id, 1)
                self.assertEqual(resp.predictions[0].prediction, 0)
                self.assertEqual(resp.predictions[1].prediction, 1)
                self.assertEqual(resp.predictions[0].class_probabilities[1], 0.4)
                self.assertEqual(resp.predictions[1].class_probabilities[1], 0.8)

    @pytest.mark.unit
    @patch('predictionapi.entities.prediction.Sentinel')
    @patch('predictionapi.entities.prediction.ProjectService', autospec = True)
    def test_smoke_reg(self, MockProjectService, MockSentinel):
        """Smoke test for prediction service for Regression. """
        svc = PredictionService(ObjectId(), ObjectId())
        result_id = str(ObjectId())
        res = json.dumps({'status': '', 'success': True, 'result_id' : result_id})
        predictions = {
            'predicted-0': [1.2, 3.0],
            'row_index': [0, 1],
            'actual': [1.0, 3.5],
            'newdata': 'YES',
            'task' : 'Regression'
        }

        project_service = MockProjectService.return_value
        project_service.get_predictions.return_value = predictions

        data = {'mimetype': 'text/plain', 'stream': 'Foo,Bar\n0,1\n1,0\n'}
        with patch.object(svc, 'tempstore') as mock_db:
            with patch.object(svc, 'client') as mock_client:
                mock_client.predict_transient.return_value = None
                mock_db.read.return_value = ('result_id', res)
                MockSentinel.return_value.master_for.return_value.blpop.return_value = ('result-key', res)
                resp = svc.predict(ObjectId(), '52ab7af8b4912910805f6f98', data)
                #self.assertTrue(mock_db.read.called)
                self.assertTrue(mock_client.predict_transient.called)
                self.assertEqual(type(resp), PredictionResponse)
                self.assertEqual(type(resp.predictions[0]), Prediction)
                self.assertEqual(resp.task, 'Regression')
                self.assertEqual(resp.predictions[0].row_id, 0)
                self.assertEqual(resp.predictions[1].row_id, 1)
                self.assertEqual(resp.predictions[0].prediction, 1.2)
                self.assertEqual(resp.predictions[1].prediction, 3.0)
                self.assertEqual(resp.predictions[0].class_probabilities, None)
                self.assertEqual(resp.predictions[1].class_probabilities, None)

    @patch('predictionapi.entities.prediction.Sentinel')
    @pytest.mark.unit
    def test_unsuccessful(self, MockSentinel):
        """Tests unsuccessful prediction. """
        svc = PredictionService(ObjectId(), ObjectId())
        res = json.dumps({'status': 'foobar', 'success': False})
        data = {'mimetype': 'text/plain', 'stream': 'Foo,Bar\n0,1\n'}
        with patch.object(svc, 'tempstore') as mock_db:
            with patch.object(svc, 'client') as mock_client:
                mock_client.predict.return_value = None
                mock_db.read.return_value = ('result-key', res)
                MockSentinel.return_value.master_for.return_value.blpop.return_value = ('result-key', res)
                with self.assertRaises(PredictionServiceUserError) as cm:
                    svc.predict(ObjectId(), '52ab7af8b4912910805f6f98', data)
                the_exception = cm.exception
                self.assertEqual(str(the_exception), 'foobar')
                #self.assertTrue(mock_db.read.called)
                self.assertTrue(mock_client.predict_transient.called)


    @patch('predictionapi.entities.prediction.Sentinel')
    @pytest.mark.unit
    def test_timeout(self, MockSentinel):
        """Tests timout when blocking for predictions. """
        svc = PredictionService(ObjectId(), ObjectId())
        res = json.dumps({'status': 'foobar', 'success': False})
        data = {'mimetype': 'text/plain', 'stream': 'Foo,Bar\n0,1\n'}
        with patch.object(svc, 'tempstore') as mock_db:
            with patch.object(svc, 'client') as mock_client:
                mock_client.predict.return_value = None
                mock_db.read.return_value = None
                MockSentinel.return_value.master_for.return_value.blpop.return_value = None
                with self.assertRaises(ValueError) as ve:
                    svc.predict(ObjectId(), '52ab7af8b4912910805f6f98', data)
                the_exception = ve.exception
                self.assertEqual(str(the_exception), 'Predictions timed out')
                #self.assertTrue(mock_db.read.called)
                self.assertTrue(mock_client.predict_transient.called)


    @patch('predictionapi.entities.prediction.Sentinel')
    @pytest.mark.unit
    def test_broker_connection_error(self, MockSentinel):
        """Tests unsuccessful connection to broker. """
        svc = PredictionService(ObjectId(), ObjectId())
        data = {'mimetype': 'text/plain', 'stream': 'Foo,Bar\n0,1\n'}
        with patch.object(svc, 'client') as mock_client:
            mock_client.client.wait_for_pong.return_value = False
            with self.assertRaises(PredictionServiceError) as pse:
                svc.predict(ObjectId(), '52ab7af8b4912910805f6f98', data)
            the_exception = pse.exception
            self.assertEqual(str(the_exception), 'Failed to connect to broker')

    @patch('predictionapi.entities.prediction.Sentinel')
    @pytest.mark.unit
    def test_job_request(self, MockSentinel):
        """Smoke tests for job request"""
        model_id = '52ab7af8b4912910805f6f98'
        svc = PredictionService(ObjectId(), ObjectId())
        data = {'mimetype': 'text/plain', 'stream': 'Foo,Bar\n0,1\n'}
        model = {}
        with patch.object(svc, 'project_service') as mock_ps:
            mock_ps.read_leaderboard_item.return_value = LB_RECORD
            job_request = svc.job_request(ObjectId(), model_id, data)
            self.assertTrue(isinstance(job_request, dict))
            self.assertEqual(job_request['predict'], 1)
            self.assertEqual(job_request['new_lid'], False)
            self.assertEqual(job_request['scoring_data'], data)

    @patch('predictionapi.entities.prediction.Sentinel')
    @pytest.mark.unit
    def test_job_request_modelid_unk(self, MockSentinel):
        """Tests job request when model_id is unknown. """
        model_id = '52ab7af8b4912910805f6f98'
        svc = PredictionService(ObjectId(), ObjectId())
        data = {'mimetype': 'text/plain', 'stream': 'Foo,Bar\n0,1\n'}
        with patch.object(svc, 'project_service') as mock_ps:
            mock_ps.read_leaderboard_item.return_value = None
            with self.assertRaises(PredictionServiceUserError) as psue:
                svc.job_request(ObjectId(), model_id, data)
            the_exception = psue.exception
            self.assertEqual(str(the_exception), 'Model {} not found'.format(model_id))
