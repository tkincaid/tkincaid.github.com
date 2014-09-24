import unittest

from bson.objectid import InvalidId, ObjectId

from MMApp.entities.roles import RoleProvider
from MMApp.entities.permissions import Permissions, Roles
from MMApp.entities.project import ProjectService
from config.test_config import db_config
from common.wrappers import database

class ProjectServiceIntegrationTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.persistent = database.new('persistent')

    @classmethod
    def tearDownClass(self):
        self.clean()

    @classmethod
    def clean(self):
        self.persistent.destroy(table='project')
        self.persistent.destroy(table='users')

    def setUp(self):
        self.clean()
        self.uid = ObjectId()
        self.pid = self.persistent.create(table='project', values={'uid': self.uid})
        self.persistent.create(table='project', values={'uid': self.uid})
        self.project_service = ProjectService(self.pid, self.uid)

    def test_set_owner_role_by_non_owner(self):
        admin_uid = '52ed139772307c28e3bf496a'

        roles = {
            '52dc081379cbafddb41ca40e' : [  'OWNER' ],
            admin_uid : [  'ADMIN' ]
        }

        self.persistent.update(table='project', condition={'_id': self.pid}, values={'roles': roles})

        self.project_service.uid = ObjectId(admin_uid)

        team_member_uid = '111111111111111111111111'
        roles = [u'OWNER']

        self.assertRaisesRegexp(ValueError, 'Only owners can designate other users as owners', self.project_service.set_role_for_team_member, team_member_uid, roles)

    def test_set_owner_role_by_owner(self):
        admin_uid = '52ed139772307c28e3bf496a'
        owner_uid = '52dc081379cbafddb41ca40e'

        roles = {
            owner_uid : [  'OWNER' ],
            admin_uid : [  'ADMIN' ]
        }

        self.persistent.update(table='project', condition={'_id': self.pid}, values={'roles': roles})

        self.project_service.uid = ObjectId(owner_uid)

        team_member_uid = '111111111111111111111111'
        roles = [u'OWNER']

        result = self.project_service.set_role_for_team_member(team_member_uid, roles)
        self.assertTrue(result)

    def test_new_team_member_has_permissions(self):
        role_provider = RoleProvider()
        owner_uid = self.persistent.create(table='users', values={'username':'owner@datarobot.com'})
        role_provider.set_roles(owner_uid, self.pid, [Roles.OWNER])

        team_member_uid = self.persistent.create(table='users', values={'username':'n00b@datarobot.com'})

        self.project_service = ProjectService(self.pid, owner_uid)
        self.project_service.add_team_member(team_member_uid)

        self.project_service = ProjectService(str(self.pid), str(team_member_uid))

    def test_get_predictions_by_dataset(self):
        dataset_id = self.persistent.create(table='metadata')

        values = {'dataset_id': str(dataset_id), 'pid': self.pid}
        values['lid'] = ObjectId()
        _id = self.persistent.create(table='predictions', values = values)

        values.pop('_id')
        values['lid'] = ObjectId()
        _id = self.persistent.create(table='predictions', values = values)

        predictions = self.project_service.get_predictions_by_dataset(dataset_id)

        self.assertTrue(predictions)
        self.assertEqual(len(predictions), 2)

    def test_get_predictions_by_dataset_does_not_include_deleted_items(self):
        dataset_id = self.persistent.create(table='metadata')

        values = {'dataset_id': str(dataset_id), 'pid': self.pid}
        lid_1 = ObjectId()
        values['lid'] = lid_1
        self.persistent.create(table='predictions', values = values)

        values.pop('_id')
        lid_2 = ObjectId()
        values['lid'] = lid_2
        values['is_deleted'] = False
        self.persistent.create(table='predictions', values = values)

        values.pop('_id')
        lid_3 = ObjectId()
        values['lid'] = lid_3
        values['is_deleted'] = True
        self.persistent.create(table='predictions', values = values)


        predictions = self.project_service.get_predictions_by_dataset(dataset_id)

        self.assertTrue(predictions)
        self.assertEqual(len(predictions), 2)

        self.assertEqual(set([p['lid'] for p in predictions]), set([lid_1, lid_2]))

    def test_get_predictions_does_not_include_deleted_items(self):

        roles = {
            str(self.uid) : [  'OWNER' ],
        }

        self.persistent.update(table='project', condition={'_id': self.pid}, values={'roles': roles})


        dataset_id = self.persistent.create(table='metadata')

        values = {'dataset_id': str(dataset_id), 'pid': self.pid}
        lid_1 = ObjectId()
        values['lid'] = lid_1
        self.persistent.create(table='predictions', values = values)

        values.pop('_id')
        lid_3 = ObjectId()
        values['lid'] = lid_3
        values['is_deleted'] = True
        self.persistent.create(table='predictions', values = values)

        predictions = self.project_service.get_predictions(lid_1, str(dataset_id))
        self.assertTrue(predictions)

        predictions = self.project_service.get_predictions(lid_3, str(dataset_id))
        self.assertFalse(predictions)


