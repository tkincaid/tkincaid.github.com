import unittest
import pytest
import copy
from mock import Mock

from bson import ObjectId
from common.entities.file_reference import FileReference

from config.test_config import db_config
from common.wrappers import database


class FileReferenceBaseCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._id = ObjectId('5223deadbeefdeadbeef1233')
        cls.storage_id = ObjectId('5223deadbeefdeadbeef1234')
        cls.project_ids = [ObjectId('5223deadbeefdeadbeef1235'),
                           ObjectId('5223deadbeefdeadbeef1236')]
        cls.reference_count = 2
        cls.uid = ObjectId('5223deadbeefdeadbeef1237')
        cls.fileref = FileReference(_id=cls._id,
                                    storage_id=cls.storage_id,
                                    project_ids=cls.project_ids,
                                    reference_count=cls.reference_count,
                                    uid=cls.uid)


class TestFileReferenceToDict(FileReferenceBaseCase):

    def test_serialization_is_sane(self):
        out = self.fileref.to_dict()
        self.assertEqual(out['storage_id'], self.storage_id)
        self.assertEqual(out['project_ids'], self.project_ids)
        self.assertEqual(out['reference_count'], self.reference_count)
        self.assertEqual(out['uid'], self.uid)

        # _id not part of to_dict
        self.assertNotIn('_id', out)

class TestFileReferenceDestroy(FileReferenceBaseCase):

    def test_cannot_destroy_fileref_with_existing_references(self):
        with self.assertRaises(ValueError):
            FileReference.destroy(self.fileref, Mock())

    def test_can_destroy_fileref_with_no_references(self):
        fileref = copy.deepcopy(self.fileref)
        fileref.remove_project_id(self.project_ids[0])
        fileref.remove_project_id(self.project_ids[1])

        self.assertEqual(fileref.reference_count, 0)

        # Just checking for not raising ValueError
        FileReference.destroy(fileref, Mock())


class TestDBInterfaceMethods(FileReferenceBaseCase):

    @classmethod
    def setUpClass(cls):
        super(TestDBInterfaceMethods, cls).setUpClass()
        cls.persistent = database.new('persistent')

    def setUp(self):
        self.persistent.destroy(table=FileReference.TABLE_NAME)
        fixture = self.fileref.to_dict()
        self.data_fixture = fixture
        self.db_backed_ref = FileReference.create(fixture['storage_id'],
            fixture['project_ids'], fixture['uid'], self.persistent)

    @pytest.mark.db
    def test_read_is_via_storage_id(self):
        result = FileReference.read(self.storage_id, self.persistent)
        self.assertEqual(result.project_ids, self.db_backed_ref.project_ids)
        self.assertEqual(result.reference_count,
                         self.db_backed_ref.reference_count)

    @pytest.mark.db
    def test_local_changes_without_update_not_stored(self):
        self.db_backed_ref.add_project_id(ObjectId('5223deadbeefdeadbeef0000'))

        result = FileReference.read(self.storage_id, self.persistent)
        self.assertNotEqual(result.reference_count,
                            self.db_backed_ref.reference_count)

    @pytest.mark.db
    def test_model_is_synced_after_update(self):
        self.db_backed_ref.add_project_id(ObjectId('5223deadbeefdeadbeef0000'))
        FileReference.update(self.db_backed_ref, self.persistent)

        result = FileReference.read(self.storage_id, self.persistent)
        self.assertEqual(result.reference_count,
                         self.db_backed_ref.reference_count)

    def test_cannot_save_reference_when_reference_count_is_off(self):
        self.db_backed_ref.reference_count = 10
        with self.assertRaises(ValueError):
            FileReference.update(self.db_backed_ref, Mock())


