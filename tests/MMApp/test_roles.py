import unittest
from mock import patch, DEFAULT
from MMApp.entities.roles import RoleProvider
from MMApp.entities.permissions import Permissions, Roles, ROLE_MAP, APP_ROLE_MAP

class RolesServiceTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_has_permissions_when_user_has_flag(self):
        role_provider = RoleProvider()

        user_permissions = {
            Permissions.CAN_LAUNCH_RSTUDIO : True
        }

        uid = '321654984536'
        pid = '655619846531'

        with patch.object(role_provider, 'get_permissions', return_value=user_permissions) as mock_get_permissions:
            result = role_provider.has_permission(uid, pid, Permissions.CAN_LAUNCH_RSTUDIO)

            self.assertTrue(result)


    def test_has_permissions_when_user_has_role(self):
        role_provider = RoleProvider()

        user_permissions = {
            Permissions.CAN_LAUNCH_RSTUDIO : True
        }

        uid = '321654984536'
        pid = '655619846531'

        with patch.multiple(role_provider, get_permissions=DEFAULT, get_roles=DEFAULT) as mock_provider:
            mock_provider['get_permissions'].return_value = {}
            mock_provider['get_roles'].return_value = [Roles.ADMIN]

            result = role_provider.has_permission(uid, pid, Permissions.CAN_LAUNCH_RSTUDIO)

            self.assertTrue(result)


    def test_get_users_by_permission(self):
        pid = '52f31d47637aba526916a3ce'

        roles = {
            '111dc081379cbafddb41ca40' : [  'OWNER' ],
            '222dc081379cbafddb41ca40' : [  'ADMIN' ],
            '333dc081379cbafddb41ca40' : [  'OBSERVER' ],
            '444dc081379cbafddb41ca40' : [  'ADMIN' ]
        }

        permissions = {
            '222dc081379cbafddb41ca40' : {
                Permissions.CAN_MANAGE_USER_ACCOUNTS: False
            }
        }

        project = {
            '_id' : '52f31d47637aba526916a3ce',
            'roles' : roles,
            'permissions' : permissions
        }

        role_provider = RoleProvider()

        with patch.object(role_provider, 'db') as mock_db:
            mock_db.read.return_value = [project]

            users = role_provider.get_uids_by_permission(pid, Permissions.CAN_MANAGE_USER_ACCOUNTS)

            self.assertEqual(len(users), 2, users)

            self.assertIn('111dc081379cbafddb41ca40', users)
            self.assertIn('444dc081379cbafddb41ca40', users)

    def test_get_users_by_roles(self):
        pid = '52f31d47637aba526916a3ce'
        roles = {
            '111dc081379cbafddb41ca40' : [  'OWNER' ],
            '222dc081379cbafddb41ca40' : [  'ADMIN' ],
            '333dc081379cbafddb41ca40' : [  'OBSERVER' ],
            '444dc081379cbafddb41ca40' : [  'ADMIN' ]
        }

        project = {
            '_id' : '52f31d47637aba526916a3ce',
            'roles' : roles
        }

        role_provider = RoleProvider()

        with patch.object(role_provider, 'db') as mock_db:
            mock_db.read.return_value = [project]


            users = role_provider.get_uids_by_roles(pid, ['ADMIN'])
            self.assertEqual(len(users), 2, users)

            self.assertIn('222dc081379cbafddb41ca40', users)
            self.assertIn('444dc081379cbafddb41ca40', users)

            users = role_provider.get_uids_by_roles(pid, ['OBSERVER'])
            self.assertEqual(len(users), 1, users)

            self.assertIn('333dc081379cbafddb41ca40', users)


    def test_describe_roles(self):
        uid = '5214d011637aba17000bbb7b'
        role_map = {
            'PERMISSION_1' : ['ROLE_1'],
            'PERMISSION_2' : ['ROLE_2'],
            'PERMISSION_3' : ['ROLE_1', 'ROLE_2'],
            'PERMISSION_4' : ['ROLE_2']
        }

        permissions = {
            'PERMISSION_5' : True
        }

        project = {
            'roles': {uid : ['ROLE_1']},
            'permissions' : {uid : permissions},
        }

        with patch.dict(ROLE_MAP, role_map, clear = True):
            effective_permissions = RoleProvider().describe_roles(uid, project)

            self.assertTrue(effective_permissions)
            self.assertEqual(len(effective_permissions.keys()), 3, effective_permissions)
            self.assertIn('PERMISSION_1', effective_permissions)
            self.assertIn('PERMISSION_3', effective_permissions)
            self.assertIn('PERMISSION_5', effective_permissions)

    def test_describe_app_roles(self):
        role_map = {
            'PERMISSION_1' : ['ROLE_1'],
            'PERMISSION_2' : ['ROLE_1'],
        }

        permissions = {
            'PERMISSION_1' : False
        }

        account = {
            'account_roles': ['ROLE_1'],
            'account_permissions' : permissions
        }

        with patch.dict(APP_ROLE_MAP, role_map, clear = True):
            effective_permissions = RoleProvider().describe_app_roles(account)

            self.assertTrue(effective_permissions)
            self.assertEqual(len(effective_permissions.keys()), 2, effective_permissions)
            self.assertFalse(effective_permissions.get('PERMISSION_1'))
            self.assertTrue(effective_permissions.get('PERMISSION_2'))




