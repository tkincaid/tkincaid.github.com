import unittest
import mock
import time

from joblib.parallel import _mk_exception as mk_joblib_exception

from ModelingMachine.engine.secure_worker import SecureWorker
from common.exceptions import TaskError
from ModelingMachine.engine.worker_request import WorkerRequest


WORKER_REQUEST = WorkerRequest({
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
        "uid": "52fc5edd21dff9eeaf624f6c",
        "time_started": time.time()
    })


class SecureWorkerTest(unittest.TestCase):
    """Some basic tests for SecureWorker.

    We patch the ZMQ context upfront.

    More elaborate tests are found in:
    IntegrationTests/test_secure_worker_unittest.py
    """

    def setUp(self):
        self.zmq_patcher = mock.patch('ModelingMachine.engine.secure_worker.zmq.Context')
        self.MockZmqContext = self.zmq_patcher.start()
        self.addCleanup(self.stopPatching)

    def stopPatching(self):
        super(SecureWorkerTest, self).tearDown()
        self.zmq_patcher.stop()

    def test_log_on_error(self):
        pipe = mock.MagicMock()
        sw = SecureWorker(WORKER_REQUEST, pipe)

        with mock.patch.object(sw, 'logger') as sw_logger:
            with mock.patch.object(sw, 'accept_job_request') as accept_job_request:
                accept_job_request.side_effect = ValueError('my side effect')
                sw.run()
                self.assertTrue(sw_logger.error.called)

    def test_error_reporting_on_system_exit(self):
        pipe = mock.MagicMock()
        sw = SecureWorker(WORKER_REQUEST, pipe)
        sw.accept_job_request = (lambda: sw.user_signal(mock.Mock(), mock.Mock()))
        with mock.patch.object(sw, 'report_error') as mock_report_error:
            sw.run()
            self.assertTrue(sw.killed)
            self.assertFalse(mock_report_error.called)

    def test_error_reporting_on_exception(self):
        pipe = mock.MagicMock()
        sw = SecureWorker(WORKER_REQUEST, pipe)
        with mock.patch.multiple(sw, accept_job_request=mock.DEFAULT,
                                 report_error=mock.DEFAULT) as mocks:
            mocks['accept_job_request'].side_effect = Exception('BOOM!')

            sw.run()

            self.assertFalse(sw.killed)
            self.assertTrue(mocks['report_error'].called)

    @mock.patch('ModelingMachine.engine.secure_worker.RequestData', autospec=True)
    def test_error_reporting_on_blueprint_error(self, mock_rd):
        """Tests if an error raised within the BP interpreter causes api.report_error. """
        pipe = mock.MagicMock()
        sw = SecureWorker(WORKER_REQUEST, pipe)
        with mock.patch.multiple(sw, accept_job_request=mock.DEFAULT, api=mock.DEFAULT) as mocks:
            mock_rd.side_effect = TaskError('internal', client_message='external', error_code=1)
            sw.run()
            self.assertTrue(mocks['api'].report_error.called)

    @mock.patch('ModelingMachine.engine.secure_worker.RequestData', autospec=True)
    def test_error_reporting_on_blueprint_joblib_error(self, mock_rd):
        """Tests if an error raised within the BP interpreter causes api.report_error. """
        pipe = mock.MagicMock()
        sw = SecureWorker(WORKER_REQUEST, pipe)
        with mock.patch.multiple(sw, accept_job_request=mock.DEFAULT, api=mock.DEFAULT) as mocks:
            joblib_task_error = mk_joblib_exception(TaskError)[0]('some message')
            mock_rd.side_effect = joblib_task_error
            sw.run()
            self.assertTrue(mocks['api'].report_error.called)
            # TODO wheck if its called indeed with a DataRobotError
