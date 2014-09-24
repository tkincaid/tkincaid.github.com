'''Tests for MetablueprintService'''
import unittest
import pytest

from mock import Mock, patch

from bson import ObjectId

from config.test_config import db_config
from common.wrappers import database

import ModelingMachine.metablueprint.base_metablueprint as basemb
from common.services.metablueprint import MetablueprintService


class TestMetablueprintServiceMethods(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.persistent = database.new('persistent')

    @classmethod
    def tearDownClass(self):
        self.persistent.destroy(table='metablueprint')

    def setUp(self):
        self.metablueprint_id = ObjectId('5223deadbeefdeadbeef0000')
        self.pid = ObjectId('5223deadbeefdeadbeef1234')
        self.uid = ObjectId('5223deadbeefdeadbeef0001')
        self.persistent.destroy(table='metablueprint')
        self.old_mb = {
            '_id': self.metablueprint_id,
            'pid': self.pid,
            'menu': {u'e27d0e503e6324730d773a2e2e4dded1': {
                u'blueprint': {
                    u'1': [[u'NUM'], [u'NI'], u'T'],
                    u'2': [[u'CAT'], [u'ORDCAT'], u'T'],
                    u'3': [[u'1', u'2'], [u'RFC e=0'], u'P']},
                u'blueprint_id': u'e27d0e503e6324730d773a2e2e4dded1',
                u'bp': 4,
                u'dataset_id': u'538a424a8bd88f18bd3d575b',
                u'features': [u'Missing Values Imputed',
                              u'Ordinal encoding of categorical variables',
                              u'RandomForest Classifier (Gini)'],
                u'diagram': 'NotWorthFaking',
                u'features_text': u'Not Worth Faking Either',
                u'icons': [1],
                u'lid': u'new',
                u'max_folds': 0,
                u'max_reps': 1,
                u'model_type': u'RandomForest Classifier (Gini)',
                u'pid': str(self.pid),
                u'reference_model': True,
                u'samplepct': 64,
                u'total_size': 158.0,
                u'uid': str(self.uid)}
            },
            'submitted_jobs': [
                u'e27d0e503e6324730d773a2e2e4dded1:'
                u'538a424a8bd88f18bd3d575b:64:1:0'
            ]
        }

    @pytest.mark.db
    def test_initialize_old_project_gets_newest_mb_and_no_flags(self):
        self.persistent.create(table=MetablueprintService.TABLE_NAME,
                               values=self.old_mb)
        cname = 'ModelingMachine.metablueprint.dev_metablueprints.Dscblueprint'
        mb_service = MetablueprintService(
            pid=self.pid, persistent=self.persistent,
            fallback_classname=cname)
        mb = mb_service.get_metablueprint(uid=self.uid,
                                          progress_inst=Mock(),
                                          reference_models=Mock(),
                                          persistent=self.persistent,
                                          tempstore=Mock())

        self.assertEqual(mb._data['classname'], cname)
        self.assertFalse(mb.flags.submitted_jobs_stored)

    @pytest.mark.db
    def test_create_metablueprint_gets_newest_mb_and_flags(self):
        cname = 'ModelingMachine.metablueprint.dev_metablueprints.Dscblueprint'
        mb_service = MetablueprintService(
            pid=self.pid, persistent=self.persistent,
            fallback_classname=cname)
        mb_service.create_metablueprint(cname)
        mb = mb_service.get_metablueprint(uid=self.uid,
                                          progress_inst=Mock(),
                                          reference_models=Mock(),
                                          persistent=self.persistent,
                                          tempstore=Mock())
        self.assertEqual(mb._data['classname'], cname)
        self.assertTrue(mb.flags.submitted_jobs_stored)

        # Tests the default flags.  Got a new one?  Add it in!
        flags_dict = mb.flags.to_dict()
        self.assertEqual(set(flags_dict.keys()),
                         set(['submitted_jobs_stored']))

    @pytest.mark.db
    def test_set_metablueprint_creates_one_if_nonexistent(self):
        '''But the flags won't be set, so be warned'''
        cname = 'ModelingMachine.metablueprint.dev_metablueprints.Dscblueprint'
        mb_service = MetablueprintService(
            pid=self.pid, persistent=self.persistent,
            fallback_classname=cname)
        mb_service.set_metablueprint(cname)
        mb = mb_service.get_metablueprint(uid=self.uid,
                                          progress_inst=Mock(),
                                          reference_models=Mock(),
                                          persistent=self.persistent,
                                          tempstore=Mock())
        self.assertEqual(mb._data['classname'], cname)
        self.assertFalse(mb.flags.submitted_jobs_stored)

    @pytest.mark.db
    def test_set_metablueprint_wont_erase_existing_flags(self):
        mb_class_base = 'ModelingMachine.metablueprint.dev_metablueprints.{}'
        cname = mb_class_base.format('Dscblueprint')
        new_name = mb_class_base.format('ETblueprint')
        mb_service = MetablueprintService(
            pid=self.pid, persistent=self.persistent,
            fallback_classname=cname)
        mb_service.create_metablueprint(cname)
        mb_service.set_metablueprint(new_name)
        mb = mb_service.get_metablueprint(uid=self.uid,
                                          progress_inst=Mock(),
                                          reference_models=Mock(),
                                          persistent=self.persistent,
                                          tempstore=Mock())
        self.assertEqual(mb._data['classname'], new_name)
        self.assertTrue(mb.flags.submitted_jobs_stored)

    @pytest.mark.db
    def test_cant_create_two_metablueprints_for_one_pid(self):
        mb_class_base = 'ModelingMachine.metablueprint.dev_metablueprints.{}'
        cname = mb_class_base.format('Dscblueprint')
        mb_service = MetablueprintService(
            pid=self.pid, persistent=self.persistent,
            fallback_classname=cname)
        mb_service.create_metablueprint(cname)
        with self.assertRaises(ValueError):
            mb_service.create_metablueprint(cname)
