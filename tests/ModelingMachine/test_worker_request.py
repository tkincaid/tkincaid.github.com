import unittest
import logging
import copy
from ModelingMachine.engine.worker_request import WorkerRequest
from ModelingMachine.engine.mocks import RequestData
from common.engine.worker_request import VertexDefinition


class TestWorkerRequest(unittest.TestCase):

    request = {
        "blueprint":
            {
                "1": [["NUM"], ["NI"], "T"],
                "3": [["1", "2"], ["GBC"], "P"],
                "2": [["CAT"], ["ORDCAT"], "T"]
            },
        "lid": "new",
        "samplepct": 100,
        "features": ["Missing Values Imputed", "Ordinal encoding of categorical variables"],
        "blueprint_id": "49e0d62fca6f42350fc65249b7c58e2e",
        "total_size": 158.0,
        "qid": "8",
        "icons": [1],
        "pid": "53052fc0637aba1f058c0de2",
        "max_reps": 1,
        "samplesize": 158.0,
        "command": "fit",
        "model_type": "Gradient Boosted Trees Classifier",
        "bp": 8,
        "max_folds": 0,
        "dataset_id":
        "53052fc0637aba1f058c0de4",
        "reference_model": True,
        "uid": "52fc5edd21dff9eeaf624f6c"
    }
    blender_request = {
            'blueprint': {'1': [['1234567890'], ['STK'], 'T'], '2': [['1'], ['GAM'], 'P']},
            'uid': '531ca1ee3e0fd11b2ecd319d',
            'blueprint_id': 'testbp_id',
            'qid': 'testqid',
            'pid': '531ca1ee3e0fd11b2ecd319e',
            'max_folds': 0,
            'dataset_id': '531ca1ee3e0fd11b2ecd319f',
            'blender': {'inputs': [
                {'blueprint': {'1': (['NUM'], ['NI'], 'T'), '2': (['1'], ['GLMB'], 'P')},
                    'dataset_id': '531ca1ee3e0fd11b2ecd319f', 'samplepct': 100, 'blender': {}},
                {'blueprint': {'1': (['NUM'], ['NI'], 'T'), '3': (['2'], ['RFC nt=10;ls=5'], 'P'), '2': (['1'], ['LR1'], 'S')},
                    'dataset_id': '531ca1ee3e0fd11b2ecd319f', 'samplepct': 100, 'blender': {}}
                ]},
            'samplepct': 64,
            'command': 'fit',
            'partitions': [[0, -1], [1, -1], [2, -1], [3, -1], [4, -1]]
        }

    def test_init_from_json(self):
        w_request = WorkerRequest(self.request)

        self.assertEqual(w_request.blueprint, self.request['blueprint'])
        self.assertEqual(w_request.pid, self.request['pid'])
        self.assertEqual(w_request.qid, self.request['qid'])
        self.assertEqual(w_request.uid, self.request['uid'])
        self.assertEqual(w_request.command, self.request['command'])

    def test_is_ping(self):
        ping_request = {'command':'ping', 'token': 'some_token'}
        w_request = WorkerRequest(ping_request)
        self.assertIsNone(w_request.preliminary_validation())
        self.assertTrue(w_request.is_ping)

    def test_input_types(self):
        req = VertexDefinition(self.request,1)
        out = req.input_types
        self.assertEqual(out, set(['N','C']))
        req = VertexDefinition(self.blender_request,1)
        out = req.input_types
        self.assertEqual(out, set(['N']))

    def test_get_inputs(self):
        stack_request = copy.deepcopy(self.request)
        stack_request['blueprint'] = {
            '1': (['NUM'],['NI'],'T'),
            '2': (['TXT'],['TM2'],'T'),
            '3': (['2'],['LASSO'],'S') }
        req = VertexDefinition(stack_request,3)
        out = req.vertex_inputs
        self.assertEqual(out, set(['T']))
        stack_request['blueprint'] = {
            '1': (['NUM'],['NI'],'T'),
            '2': (['TXT'],['TM2'],'T'),
            '3': (['1','2'],['LASSO'],'S') }
        req = VertexDefinition(stack_request,3)
        out = req.vertex_inputs
        self.assertEqual(out, set(['N','T']))

    def test_validate(self):
        data = RequestData(1,2,3)

        data.partition = {'reps':5,'folds':5,'seed':1234,'holdout_pct':20,'total_size':1000}
        req = WorkerRequest(self.request)
        self.assertEqual(req.samplepct, 100)
        req.validate(data)
        self.assertEqual(req.samplepct, 100.0)

        data.partition = {'reps':5,'folds':5,'seed':1234,'holdout_pct':10,'total_size':1000}
        req = WorkerRequest(self.request)
        self.assertEqual(req.samplepct, 100)
        req.validate(data)
        self.assertEqual(req.samplepct, 100.0)

        data.partition = {'reps':10,'folds':1,'seed':1234,'holdout_pct':10,'total_size':1000}
        req = WorkerRequest(self.request)
        self.assertEqual(req.samplepct, 100)
        req.validate(data)
        self.assertEqual(req.samplepct, 100.0)




if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logger.addHandler(console)

    unittest.main()
