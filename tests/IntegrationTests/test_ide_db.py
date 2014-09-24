import unittest

from mock import patch, DEFAULT
from bson import ObjectId

from common.wrappers import database
from MMApp.entities.ide import IdeService, IdeSetupStatus

class IdeServiceIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.persistent = database.new('persistent')
        self.tempstore = database.new('tempstore')

        self.broker_client_patcher = patch('MMApp.entities.ide.IdeBrokerClient', autospec = True)
        self.broker_client_patcher.start()

    @classmethod
    def tearDownClass(self):
        self.clean()
        self.broker_client_patcher.stop()

    @classmethod
    def clean(self):
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='users')

    def setUp(self):
        self.clean()
        self.project = {'uid': ObjectId(), 'default_dataset_id': ObjectId(), 'pid' : ObjectId()}
        self.create_project()


    def create_project(self):
        self.project['pid'] = self.persistent.create(values= self.project, table='project')
        self.persistent.create(values= {
            'file_id': self.project['default_dataset_id'],
            'pid': self.project['pid'],
            'originalName': 'file name',
            'files' : []
        }, table='metadata')

    #FIXME: the other integration tests have to be updated to not destroy the new redis keys before turning these tests back on
    @patch('MMApp.entities.ide.logger')
    def skip_test_quick_requests_to_setup(self, mock_logger):
        service = IdeService(self.project['uid'], self.project['_id'])

        self.assertEqual(service.setup().status, IdeSetupStatus.STARTED)

        for i in xrange(10):
            result = service.setup()
            self.assertEqual(result.status, IdeSetupStatus.STARTED)

        #the logger is used to confirm this function is working correctly, sorry for any inconvenience
        self.assertEqual(mock_logger.debug.call_count, 21)

    def skip_test_frequent_error_and_setup_requests(self):
        service = IdeService(self.project['uid'], self.project['_id'])

        setup_status = service.setup()

        for i in xrange(10):
            # Report error and setup multiple times, service should't flinch
            service.report_error()
            current_status = service.setup()
            #self.assertGreaterEqual(set(result.to_dict().keys()), set('status', 'remove', 'key', 'timestamp'))
            print setup_status.to_dict()
            print current_status.to_dict()
            self.assertEqual(setup_status.to_dict(), current_status.to_dict())

if __name__ == '__main__':
    unittest.main()
