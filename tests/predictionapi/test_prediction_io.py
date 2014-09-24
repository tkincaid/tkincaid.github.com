import unittest
import json

from bson import ObjectId
from itertools import repeat

from predictionapi.prediction_io import ClassificationPrediction
from predictionapi.prediction_io import RegressionPrediction
from predictionapi.prediction_io import PredictionResponse
from predictionapi.prediction_io import JsonResponseSerializer
from predictionapi.prediction_io import TemplateJsonResponseSerializer
from predictionapi.prediction_io import JSONUserTopkRequest


class JSONPredictionResponseSerializerTest(unittest.TestCase):

    Serializer = JsonResponseSerializer

    def test_json_serializer_cls(self):
        n = 1000
        data = zip(range(n), repeat(0), repeat({'0': 1.0, '1': 0.0}))
        predictions = map(ClassificationPrediction._make, data)
        model_id = str(ObjectId())
        response = PredictionResponse(model_id, 'Binary', 1000, predictions)
        serializer = self.Serializer('v1')
        enc_resp = serializer.serialize_success(response)
        resp = json.loads(enc_resp)
        self.assertEqual(resp['code'], 200)
        self.assertEqual(resp['model_id'], model_id)
        self.assertEqual(resp['execution_time'], 1000)
        self.assertEqual(resp['task'], 'Binary')
        self.assertEqual(resp['status'], '')
        self.assertDictEqual(resp['predictions'][0], {'row_id': 0, 'prediction': 0,
                                                      'class_probabilities': {'0': 1.0,
                                                                              '1': 0.0}})
        self.assertDictEqual(resp['predictions'][-1], {'row_id': n - 1, 'prediction': 0,
                                                      'class_probabilities': {'0': 1.0,
                                                                              '1': 0.0}})

    def test_json_serializer_reg(self):
        n = 1000
        data = zip(range(n), repeat(0))
        predictions = map(RegressionPrediction._make, data)
        model_id = str(ObjectId())
        response = PredictionResponse(model_id, 'Regression', 1000, predictions)
        serializer = self.Serializer('v1')
        enc_resp = serializer.serialize_success(response)
        resp = json.loads(enc_resp)
        self.assertEqual(resp['code'], 200)
        self.assertEqual(resp['model_id'], model_id)
        self.assertEqual(resp['execution_time'], 1000)
        self.assertEqual(resp['task'], 'Regression')
        self.assertEqual(resp['status'], '')
        self.assertDictEqual(resp['predictions'][0], {'row_id': 0, 'prediction': 0})
        self.assertDictEqual(resp['predictions'][-1], {'row_id': n - 1, 'prediction': 0})


class TemplateJSONPredictionResponseSerializerTest(JSONPredictionResponseSerializerTest):

    Serializer = TemplateJsonResponseSerializer


class UserTopkRequestTest(unittest.TestCase):

    def test_generate_request(self):
        stream = json.dumps({'user_id': [1, 2, 3],
                             'known_items': True})
        req = JSONUserTopkRequest(stream, None)
        self.assertEqual(req.user_id, [1, 2, 3])
        self.assertEqual(req.known_items, True)
        self.assertEqual(req.n_items, 20)
        self.assertEqual(req.threshold, None)
