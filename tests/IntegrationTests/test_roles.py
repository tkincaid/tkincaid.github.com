import unittest
import config.test_config as config
from common.wrappers import database
from MMApp.entities.roles import RoleProvider
from MMApp.entities.permissions import Roles, Permissions, AccountRoles

class RoleServiceTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.db = database.new('persistent')

    @classmethod
    def tearDownClass(self):
        self.clean()

    @classmethod
    def clean(self):
        self.db.destroy(table='project')
        self.db.destroy(table='users')

    def setUp(self):
        self.clean()
        self.uid = database.ObjectId(None)
        self.uid = str(self.uid)

    def test_get_roles(self):
        pid = self.db.create(table='project', values={'roles': {self.uid: [Roles.ADMIN]}})

        role_provider = RoleProvider()
        access = role_provider.has_permission(self.uid, pid, Permissions.CAN_LAUNCH_RSTUDIO)
        self.assertTrue(access)

        access = role_provider.has_permission(self.uid, pid, 'DOES NOT EXIST')
        self.assertFalse(access)

    def test_get_permissions(self):
        pid = self.db.create(table='project', values={'permissions': {self.uid: {
            Permissions.CAN_LAUNCH_RSTUDIO : True
            }}})

        role_provider = RoleProvider()
        access = role_provider.has_permission(self.uid, pid, Permissions.CAN_LAUNCH_RSTUDIO)
        self.assertTrue(access)

        access = role_provider.has_permission(self.uid, pid, 'DOES NOT EXIST')
        self.assertFalse(access)


    def test_set_roles(self):
        pid = self.db.create(table='project', values={'name': 'untitled'})
        role_provider = RoleProvider()

        roles = [Roles.ADMIN, Roles.OBSERVER]
        role_provider.set_roles(self.uid, pid, roles)

        actual_roles = role_provider.get_roles(self.uid, pid)

        self.assertEquals(set(roles), set(actual_roles))

    def test_set_roles_does_not_overwrite(self):
        pid = self.db.create(table='project', values={'name': 'untitled'})
        role_provider = RoleProvider()

        roles = [Roles.OWNER, Roles.ADMIN, Roles.OBSERVER]
        role_provider.set_roles(self.uid, pid, roles)

        second_uid = str(database.ObjectId(None))
        role_provider.set_roles(second_uid, pid, roles[:2])

        actual_roles = role_provider.get_roles(self.uid, pid)
        self.assertEquals(set(roles), set(actual_roles))

        actual_roles = role_provider.get_roles(second_uid, pid)
        self.assertEquals(set(roles[:2]), set(actual_roles))

    def test_set_permissions(self):
        pid = self.db.create(table='project', values={'name': 'untitled'})
        role_provider = RoleProvider()

        expected_permissions = {
            Permissions.CAN_LAUNCH_RSTUDIO: True
        }

        role_provider.set_permissions(self.uid, pid, expected_permissions)

        actual_permissions = role_provider.get_permissions(self.uid, pid)
        self.assertEquals(set(expected_permissions), set(actual_permissions))

    def test_set_permissions_does_not_overwrite(self):
        pid = self.db.create(table='project', values={'name': 'untitled'})
        role_provider = RoleProvider()

        user_1_permissions = {
            Permissions.CAN_LAUNCH_RSTUDIO: True
        }

        role_provider.set_permissions(self.uid, pid, user_1_permissions)

        user_2_permissions = {
            Permissions.CAN_MANAGE_MODELS: True
        }

        second_uid = str(database.ObjectId(None))
        role_provider.set_permissions(second_uid, pid, user_2_permissions)

        actual_permissions = role_provider.get_permissions(self.uid, pid)
        self.assertEquals(set(user_1_permissions), set(actual_permissions))

        actual_permissions = role_provider.get_permissions(second_uid, pid)
        self.assertEquals(set(user_2_permissions), set(actual_permissions))

    def test_get_roles_match(self):
        pid = self.db.create(table='project', values={'name': 'untitled'})
        role_provider = RoleProvider()

        roles = [Roles.OWNER, Roles.ADMIN, Roles.OBSERVER]
        role_provider.set_roles(self.uid, pid, roles)

        p_roles = role_provider.read('project', pid,
                                     'roles').get(str(self.uid), [])
        u_roles = role_provider.read('users', self.uid,
                                     'roles').get(str(pid), [])

        self.assertTrue(p_roles)
        self.assertEqual(p_roles, u_roles)

    def test_has_permission(self):
        pid = self.db.create(table='project', values={'name': 'untitled'})
        uid = self.db.create(table='users', values={'username':'test@datarobot.com'})

        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, Roles.DATA_SCIENTIST)

        result = role_provider.has_permission(uid, pid, Permissions.CAN_VIEW)

        self.assertTrue(result)

    def test_delete_roles(self):
        pid = self.db.create(table='project', values={'name': 'untitled'})
        uid = self.db.create(table='users', values={'username':'test@datarobot.com'})

        role_provider = RoleProvider()
        role_provider.set_roles(uid, pid, Roles.DATA_SCIENTIST)

        roles = role_provider.read('project', pid, 'roles')
        self.assertIn(str(uid), roles)

        roles = role_provider.read('users', uid, 'roles')
        self.assertIn(str(pid), roles)

        role_provider.delete_roles(uid, pid)

        roles = role_provider.read('project', pid, 'roles')
        self.assertNotIn(str(uid), roles)

        roles = role_provider.read('users', uid, 'roles')
        self.assertNotIn(str(pid), roles)

    def test_account_permissions(self):
        uid = self.db.create(table='users', values={'username':'test@datarobot.com'})

        role_provider = RoleProvider()

        actual_permissions = role_provider.get_account_permissions(uid)
        self.assertFalse(actual_permissions)

        permissions = {
            Permissions.GRAYBOX_ENABLED : True,
            Permissions.IS_OFFLINE_QUEUE_ENABLED:True
        }

        role_provider.set_account_permissions(uid, permissions)

        actual_permissions = role_provider.get_account_permissions(uid)

        self.assertTrue(actual_permissions)
        self.assertEqual(actual_permissions.get(Permissions.GRAYBOX_ENABLED), True)
        self.assertEqual(actual_permissions.get(Permissions.IS_OFFLINE_QUEUE_ENABLED), True)

        self.assertTrue(role_provider.account_has_permission(uid, Permissions.GRAYBOX_ENABLED))
        self.assertTrue(role_provider.account_has_permission(uid, Permissions.IS_OFFLINE_QUEUE_ENABLED))

        permissions = {
            Permissions.IS_OFFLINE_QUEUE_ENABLED:False
        }

        role_provider.set_account_permissions(uid, permissions)

        actual_permissions = role_provider.get_account_permissions(uid)

        self.assertTrue(actual_permissions)
        self.assertEqual(actual_permissions.get(Permissions.GRAYBOX_ENABLED), True)
        self.assertEqual(actual_permissions.get(Permissions.IS_OFFLINE_QUEUE_ENABLED), False)

        self.assertTrue(role_provider.account_has_permission(uid, Permissions.GRAYBOX_ENABLED))
        self.assertFalse(role_provider.account_has_permission(uid, Permissions.IS_OFFLINE_QUEUE_ENABLED))

    def test_account_roles(self):

        uid = self.db.create(table='users', values={'username':'test@datarobot.com'})

        role_provider = RoleProvider()

        actual_roles = role_provider.get_account_roles(uid)
        self.assertFalse(actual_roles)

        role_provider.set_account_roles(uid, AccountRoles.USER)

        actual_roles = role_provider.get_account_roles(uid)

        self.assertTrue(actual_roles)
        self.assertEqual([AccountRoles.USER], actual_roles)
        self.assertTrue(role_provider.account_has_permission(uid, Permissions.GRAYBOX_ENABLED))
        self.assertFalse(role_provider.account_has_permission(uid, Permissions.CAN_MANAGE_APP_USERS))

        role_provider.set_account_roles(uid, AccountRoles.APP_USER_MANAGER)

        actual_roles = role_provider.get_account_roles(uid)

        self.assertTrue(actual_roles)
        self.assertEqual([AccountRoles.USER, AccountRoles.APP_USER_MANAGER], actual_roles)
        self.assertTrue(role_provider.account_has_permission(uid, Permissions.CAN_MANAGE_APP_USERS))
