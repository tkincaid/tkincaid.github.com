from config.engine import EngConfig
from common.api.api_client import APIClient
from integration_test_base import IntegrationTestBase
from safeworkercode.docker_client import DockerClient

class TestUserModels(IntegrationTestBase):

    @classmethod
    def setUpClass(self):
        super(TestUserModels, self).setUpClass()
        TestUserModels.pid = None

    def setUp(self):
        self.set_dataset('kickcars-sample-200.csv', 'IsBadBuy')
        if not TestUserModels.pid:
            TestUserModels.pid = self.create_project()
            TestUserModels.uid = self.registered_user['uid']

        self.pid = TestUserModels.pid

    def get_ipython_notebook_template(self):
        notebook_path = self.path_to_test_file('ipython-notebook.py')
        with open(notebook_path, 'r') as f:
            return f.read()

    def test_user_model(self):
        self.submit_user_model(with_error = False)

    def test_user_model_error(self):
        self.submit_user_model(with_error= True)

    def submit_user_model(self, with_error):

        notebook = self.get_ipython_notebook_template()

        if with_error:
            notebook = notebook.replace('return self', 'return UNDEFINED')

        user_model_request = {
            'samplepct': 64,
            'uid': self.uid,
            'modelsource': notebook,
            'pid': self.pid,
            'classname': 'CustomModel',
            'key': 'Jo8Vh-MK5O90Uw==',
            'model_type': 'IPython Model 1'
        }

        user_model_q_item = self.create_user_model_job(user_model_request)
        # Make sure the leaderboard returns our user model
        leaderboard = self.get_models(self.app, self.pid)
        self.assertTrue(leaderboard)
        user_models = [l for l in leaderboard if str(l['qid']) == str(user_model_q_item['qid'])]
        model = user_models[0] if user_models else None
        self.assertTrue(model, 'User Model not found in leaderboard')

        if with_error:
            # Model has error
            self.assertTrue(model.get('build_error'), 'The user model was not marked as error: {}'.format(model))
            model = self.get_model(self.pid, lid = model.get('_id'))

            # Model has stacktrace with no DataRobot code
            message = model.get('logs') and model['logs'][0].get('message')
            self.assertTrue(message, 'The model does not have any logs saved: {}'.format(model))
            self.assertNotIn('ModelingMachine', message,
                'Stacktrace contains DataRobot code files')
            self.assertIn("global name 'UNDEFINED' is not defined", message,
                'Model does not contain the user stack trace: {}'.format(message))

        else:
            self.assertFalse(model.get('build_error'), 'The user model was marked as error: {}'.format(model))


    def create_user_model_job(self, user_model_request):
        pid = self.pid

        self.wait_for_stage(self.app, pid, 'modeling')

        self.set_q_with_n_items(pid, 0, self.app)

        api_client = APIClient(EngConfig['WEB_API_LOCATION'])
        api_client.create_queue_item(user_model_request)

        # Make sure our user model is in the q
        q = self.get_q(pid)

        items =  filter(lambda item: 'IPython' in item['model_type'] , q)
        self.assertTrue(items, 'User model not found in q: {}'.format(q))
        user_model_q_item = items[0] if items else None
        self.assertIsNotNone(user_model_q_item)

        self.service_queue(pid)
        self.wait_for_q_item_to_complete(self.app, pid, str(user_model_q_item['qid']), 15)
        return user_model_q_item

    def does_secure_container_image_exist(self, q_item):
        docker_client = DockerClient()
        name = '{0}-{1}'.format(q_item['pid'], q_item['qid'])
        return docker_client.find_image_by_name(name)


