import unittest
import mock

from common.engine.error_reporter import ErrorReporter
from common.exceptions import GENERIC_ERROR_MESSAGE
from common.exceptions import GENERIC_WHITEBOX_ERROR_MESSAGE
from common.exceptions import WHITEBOX_ERROR_MESSAGE
from common.exceptions import DataRobotError
from common.exceptions import ModelingMachineError
from common.exceptions import TaskError
from common.exceptions import ErrorCode


class ErrorReporterTest(unittest.TestCase):

    def test_smoke(self):
        error = DataRobotError('internal',
                               client_message='external',
                               error_code=123)
        mock_progress = mock.Mock()
        ErrorReporter.report(mock_progress, error)
        self.assertTrue(mock_progress.error.called)
        mock_progress.error.assert_called_once_with(
            GENERIC_ERROR_MESSAGE.format(error_code=123))

    def test_smoke_no_error_code(self):
        error = DataRobotError('internal',
                               client_message='external')
        mock_progress = mock.Mock()
        ErrorReporter.report(mock_progress, error)
        self.assertTrue(mock_progress.error.called)
        mock_progress.error.assert_called_once_with(
            GENERIC_ERROR_MESSAGE.format(error_code=ErrorCode.GENERIC_ERROR))

    def test_mm_error(self):
        error = ModelingMachineError('internal',
                                     client_message='external',
                                     error_code=123)
        mock_progress = mock.Mock()
        ErrorReporter.report(mock_progress, error)
        self.assertTrue(mock_progress.error.called)
        mock_progress.error.assert_called_once_with(
            GENERIC_ERROR_MESSAGE.format(error_code=123))

    def test_task_error(self):
        error = TaskError('internal', client_message='external',
                          error_code=123)
        mock_progress = mock.Mock()
        ErrorReporter.report(mock_progress, error)
        self.assertTrue(mock_progress.error.called)
        mock_progress.error.assert_called_once_with(
            WHITEBOX_ERROR_MESSAGE.format(client_message='external',
                                          error_code=123))

    def test_task_error_no_msg(self):
        error = TaskError('internal', client_message='',
                          error_code=123)
        mock_progress = mock.Mock()
        ErrorReporter.report(mock_progress, error)
        self.assertTrue(mock_progress.error.called)
        mock_progress.error.assert_called_once_with(
            GENERIC_WHITEBOX_ERROR_MESSAGE.format(error_code=123))

