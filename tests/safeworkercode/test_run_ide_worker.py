import unittest
from mock import patch

from safeworkercode.run_ide_worker import IDEWorkers

class TestRunIdeWorker(unittest.TestCase):

    def setUp(self):
        self.patchers = []
        zmq_patch = patch('common.broker.workers.zmq')
        self.patchers.append(zmq_patch)
        zmq_patch.start()
        self.addCleanup(self.stopRunIdeWorker)

    def stopRunIdeWorker(self):
        super(TestRunIdeWorker, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    @patch('safeworkercode.run_ide_worker.APIClient')
    def test_get_available_cpu(self, mock_api):
        with patch('safeworkercode.safe_workers.multiprocessing') as mock_multiprocessing:
            #Arrange
            mock_multiprocessing.cpu_count.return_value = 3
            workers = IDEWorkers(None)
            workers.current_cpu = -1
            self.assertEqual(workers.cpu_count, 3)

            #Act
            workers.get_available_cpu()

            #Assert
            self.assertEqual(workers.current_cpu, 0)

            #Act
            workers.get_available_cpu()
            workers.get_available_cpu()

            #Assert
            self.assertEqual(workers.current_cpu, 2)


            #Act
            workers.get_available_cpu()
            #Assert
            self.assertEqual(workers.current_cpu, 0)


            #Act
            workers.get_available_cpu()
            #Assert
            self.assertEqual(workers.current_cpu, 1)

    @patch('safeworkercode.safe_workers.multiprocessing')
    @patch('safeworkercode.run_ide_worker.APIClient')
    def test_register(self, mock_api, mock_multiprocessing):
        mock_api.return_value.register_worker.return_value = "1"
        workers = IDEWorkers(None)
        self.assertIsNone(workers.register())
        self.assertEqual(workers.worker_id, "1")



