import unittest
import json
from mock import patch
from config.engine import EngConfig
from common.wrappers import database
from MMApp.entities.user import UserModel
from MMApp.entities.admin import AdminAccessError

class TestAdminBlueprint(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.stopPatching)
        self.app = None
        with patch.dict(EngConfig, {'TEST_MODE': True }, clear = False):
            import MMApp.app
            self.app = MMApp.app.app.test_client()

        self.patchers = []
        get_user_session_patch = patch('MMApp.app_admin.get_user_session')
        self.patchers.append(get_user_session_patch)
        self.mock_get_user_session = get_user_session_patch.start()

        mock_user = {
            'uid': database.ObjectId(None),
            'username':'project-owner@datarobot.com'
        }

        self.mock_get_user_session.return_value = UserModel(**mock_user)

        user_flags_admin_patch = patch('MMApp.app_admin.UserFlagsAdmin')
        self.patchers.append(user_flags_admin_patch)
        self.MockUserFlagsAdmin = user_flags_admin_patch.start()

        user_service_patch = patch('MMApp.app_admin.UserService')
        self.patchers.append(user_service_patch)
        self.MockUserservice = user_service_patch.start()
        self.MockUserservice.return_value.get_account.return_value = mock_user

    def stopPatching(self):
        super(TestAdminBlueprint, self).tearDown()
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    def test_save_profile(self):
        user = {
            'max_workers' : 10,
            'permissions' : {
                'PERMISSION_1' : True,
                'PERMISSION_2' : False
            }

        }

        response = self.app.post('/users/{}'.format(database.ObjectId(None)),
            content_type='application/json', data=json.dumps(user))

        self.assertEqual(response.status_code, 200)

    def test_get_permissions_lists(self):
        response = self.app.get('/users/permissions')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['permissions'])

    def test_activate_account(self):
        user = {
            'activated' : 0
        }

        response = self.app.post('/users/{}/activate'.format(database.ObjectId(None)),
            content_type = ' application/json', data = json.dumps(user))

        self.assertEqual(response.status_code, 200)

    def test_search_users(self):
        keyword = 'hello'
        admin_service = self.MockUserFlagsAdmin.return_value

        users = [
            UserModel(username='user1'),
            UserModel(username='user2'),
            UserModel(username='user3'),
        ]

        admin_service.search_users_by_name_and_username.return_value = users

        response = self.app.get('/users/search/{}'.format(keyword))

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertEqual(len(response_data['users']), len(users))

    def test_search_users_without_permissions(self):
        admin_service = self.MockUserFlagsAdmin.return_value
        admin_service.search_users_by_name_and_username.side_effect = AdminAccessError()

        response = self.app.get('/users/search/{}'.format('hi'))

        self.assertEqual(response.status_code, 403)