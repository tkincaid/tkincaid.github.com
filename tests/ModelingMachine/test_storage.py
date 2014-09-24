import sys
import os

mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))

import common.storage as storage
from common.storage import FileStorageClient
from common.storage import LocalStorage

import unittest
import tempfile
import time
import logging
import shutil

from mock import Mock, patch, call, DEFAULT

class StorageTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #TODO: patch EngConfig
        storage.EngConfig['FILE_STORAGE_TYPE'] = "local"
        storage.EngConfig['FILE_STORAGE_PREFIX'] = "test_storage/"
        cls.LOCAL_FILE_STORAGE_DIR = '/tmp/test_storage_lfsd'
        storage.EngConfig['LOCAL_FILE_STORAGE_DIR'] = cls.LOCAL_FILE_STORAGE_DIR
        storage.EngConfig['S3_BUCKET'] = "dataroboteast"
        storage.EngConfig['CACHE_DIR'] = "/tmp/test_storage_cache"
        storage.EngConfig['CACHE_LIMIT'] = 10*1024*1024

        cls.file_name = "test"
        cls.file_contents = "test"

        cls.file_local = storage.FileObject(cls.file_name, storage_type="local")

    @classmethod
    def tearDownClass(self):
        self.file_local.delete()

    def setUp(self):
        self.file_local.delete()

        self.local_file = tempfile.NamedTemporaryFile(mode='w',delete=False)
        self.local_file.file.write(self.file_contents)
        self.local_file.close()

        self.fourmb = "/tmp/fourmb_file"
        with open(self.fourmb, "wb") as f:
            f.seek((4 * 1024 * 1024) - 1)
            f.write('\0')

    def tearDown(self):
        try:
            self.local_file.unlink(self.local_file.name)
            os.remove(self.fourmb)
            shutil.rmtree(self.LOCAL_FILE_STORAGE_DIR)
            shutil.rmtree(storage.EngConfig['CACHE_DIR'])
        except:
            pass


    def addTestData(self):
        pass


    @patch('common.storage.os.path.isdir')
    def test_race_condition_when_creating_dirs(self, MockIsDir):
        self.race_condition_path = os.path.join(self.LOCAL_FILE_STORAGE_DIR, 'race-condition-dir')
        os.makedirs(self.race_condition_path)


        # Create race condition, our process thinks the file hasn't been created
        MockIsDir.return_value = False

        # No exception
        storage = LocalStorage(self.race_condition_path)

        self.assertIsNotNone(storage)

    def test_exists(self):
        # self.assertFalse(self.file_s3.exists())

        # rv = self.file_s3.put(self.local_file.name)
        # self.assertTrue(rv)

        # #sleep to give s3 time to become consistent
        # #how can this test be made deterministic?
        # time.sleep(2)
        # self.assertTrue(self.file_s3.exists())

        ### Local

        self.assertFalse(self.file_local.exists())

        rv = self.file_local.put(self.local_file.name)
        self.assertTrue(rv)

        self.assertTrue(self.file_local.exists())

    def test_get(self):
        # #check that the file doesn't already exist
        # self.assertFalse(self.file_s3.exists())

        # #put a file into storage
        # rv = self.file_s3.put(self.local_file.name)
        # #check that the file was sent to storage
        # self.assertTrue(rv)

        # #create a temporary local file
        # local_file = tempfile.NamedTemporaryFile(mode='w',delete=False)
        # local_file.close()

        # #overwrite the temporary local file with the file from storage
        # self.file_s3.get(local_file.name)

        # #get the contents of the temporary local file
        # with open(local_file.name) as f:
        #     contents = f.read()

        # #check that the content of the temporary local file matches the file that was put into storage
        # self.assertEqual(contents, self.file_contents)

        # #remove the temporary local file
        # local_file.unlink(local_file.name)

        ### Local

        #check that the file doesn't already exist
        self.assertFalse(self.file_local.exists())

        #put a file into storage
        rv = self.file_local.put(self.local_file.name)
        #check that the file was sent to storage
        self.assertTrue(rv)

        #create a temporary local file
        local_file = tempfile.NamedTemporaryFile(mode='w',delete=False)
        local_file.close()

        #overwrite the temporary local file with the file from storage
        self.file_local.get(local_file.name)

        #get the contents of the temporary local file
        with open(local_file.name) as f:
            contents = f.read()

        #check that the content of the temporary local file matches the file that was put into storage
        self.assertEqual(contents, self.file_contents)

        #remove the temporary local file
        local_file.unlink(local_file.name)

    def test_put(self):
        # self.assertFalse(self.file_s3.exists())

        # rv = self.file_s3.put(self.local_file.name)
        # self.assertTrue(rv)

        # #sleep to give s3 time to become consistent
        # #how can this test be made deterministic?
        # time.sleep(2)
        # self.assertTrue(self.file_s3.exists())

        ### Local

        self.assertFalse(self.file_local.exists())

        rv = self.file_local.put(self.local_file.name)
        self.assertTrue(rv)

        self.assertTrue(self.file_local.exists())

    def test_delete(self):
        # rv = self.file_s3.delete()
        # self.assertTrue(rv)

        # rv = self.file_s3.put(self.local_file.name)
        # self.assertTrue(rv)

        # time.sleep(2)
        # self.assertTrue(self.file_s3.exists())

        # rv = self.file_s3.delete()
        # self.assertTrue(rv)

        ### Local

        rv = self.file_local.delete()
        self.assertTrue(rv)

        rv = self.file_local.put(self.local_file.name)
        self.assertTrue(rv)

        self.assertTrue(self.file_local.exists())

        rv = self.file_local.delete()
        self.assertTrue(rv)

    def test_move(self):
        self.assertFalse(self.file_local.exists())

        rv = self.file_local.move(self.local_file.name)
        self.assertTrue(rv)

        self.assertTrue(self.file_local.exists())
        self.assertFalse(os.path.isfile(self.local_file.name))

    def test_cache(self):
        f1 = storage.FileObject("file1")
        #cache defaults to None when a FileObject is using local storage, so add the cache
        f1.cache = storage.Cache()

        f2 = storage.FileObject("file2")
        f2.cache = storage.Cache()

        f3 = storage.FileObject("file3")
        f3.cache = storage.Cache()

        rv = f1.put(self.fourmb)
        self.assertTrue(rv)

        self.assertTrue(f1.exists())
        self.assertTrue(f1.cache.exists(f1.name))

        rv = f2.put(self.fourmb)
        self.assertTrue(rv)

        self.assertTrue(f2.exists())
        self.assertTrue(f2.cache.exists(f2.name))

        #the cache is limited to 10MB, there shouldn't be enough room for the third four-mb file
        rv = f3.put(self.fourmb)
        self.assertTrue(rv)

        self.assertTrue(f3.exists())
        self.assertTrue(f3.cache.exists(f3.name))
        self.assertFalse(f1.cache.exists(f1.name) and f2.cache.exists(f2.name))

        f1.delete()
        f2.delete()
        f3.delete()

        self.assertFalse(f1.cache.exists(f1.name) or f2.cache.exists(f2.name) or f3.cache.exists(f3.name))
        self.assertFalse(f1.exists() or f2.exists() or f3.exists())

    def test_s3_exists(self):
        boto = Mock()
        sys.modules['boto'] = boto
        connection = boto.connect_s3.return_value
        bucket = connection.lookup.return_value

        s3fo = storage.FileObject("test", storage_type="s3")

        bucket.get_key.return_value = False
        self.assertFalse(s3fo.exists())

        bucket.get_key.return_value = True
        self.assertTrue(s3fo.exists())

    def test_s3_get_file_does_not_exist(self):
        boto = Mock()
        sys.modules['boto'] = boto
        connection = boto.connect_s3.return_value
        bucket = connection.lookup.return_value

        s3fo = storage.FileObject("test", storage_type="s3")

        bucket.get_key.return_value = None
        self.assertFalse(s3fo.get("local_filename"))

    def test_s3_get_file_error(self):
        boto = Mock()
        sys.modules['boto'] = boto
        connection = boto.connect_s3.return_value
        bucket = connection.lookup.return_value
        key = bucket.get_key.return_value

        s3fo = storage.FileObject("test", storage_type="s3")

        self.assertTrue(s3fo.get("local_filename"))

        key.get_contents_to_filename.side_effect = Exception()
        self.assertFalse(s3fo.get("local_filename"))

    def test_s3_put(self):
        boto = Mock()
        sys.modules['boto'] = boto
        connection = boto.connect_s3.return_value
        bucket = connection.lookup.return_value
        key = bucket.new_key.return_value

        s3fo = storage.FileObject("test", storage_type="s3")

        self.assertTrue(s3fo.put("local_filename"))

        key.set_contents_from_filename.side_effect = Exception()
        self.assertFalse(s3fo.put("local_filename"))

    def test_s3_delete(self):
        boto = Mock()
        sys.modules['boto'] = boto
        connection = boto.connect_s3.return_value
        bucket = connection.lookup.return_value

        s3fo = storage.FileObject("test", storage_type="s3")

        self.assertTrue(s3fo.delete())

        bucket.delete_key.side_effect = Exception()
        self.assertFalse(s3fo.delete())

    def test_s3_url(self):
        boto = Mock()
        sys.modules['boto'] = boto
        connection = boto.connect_s3.return_value
        bucket = connection.lookup.return_value
        key = bucket.get_key.return_value
        key.generate_url.return_value = "url"

        s3fo = storage.FileObject("test", storage_type="s3")

        self.assertEqual(s3fo.url(), key.generate_url.return_value)

        key.generate_url.side_effect = Exception()
        self.assertIsNone(s3fo.url())

    def test_fsc_get_file(self):
        pid = "1"
        fsc = FileStorageClient()
        with patch('common.storage.FileTransaction') as mock_fo:
                with patch.multiple(fsc, _get_fileobj_with_retry=DEFAULT) as mocks:
                    fo = mock_fo.return_value
                    mocks['_get_fileobj_with_retry'].return_value = "name"
                    self.assertEqual(fsc._get_file("filename", pid=pid), "name")
                    self.assertTrue(fo.exists.called)

    def test_fsc_get_file_not_exist(self):
        pid = "1"
        fsc = FileStorageClient()
        with patch('common.storage.FileTransaction') as mock_fo:
                with patch.multiple(fsc, _get_fileobj_with_retry=DEFAULT) as mocks:
                    fo = mock_fo.return_value
                    fo.exists.return_value = False
                    self.assertIsNone(fsc._get_file("filename", pid=pid))

    @patch('common.storage.tempfile')
    @patch('common.storage.os')
    def test_fsc_get_file_with_retry(self, mock_os, mock_tempfile):
        with patch.multiple(self.file_local, get=DEFAULT) as mocks:
            fsc = FileStorageClient()
            mock_tempfile.NamedTemporaryFile.return_value.name = "local_filename"
            self.assertEqual(fsc._get_fileobj_with_retry(self.file_local), "local_filename")

    @patch('common.storage.time')
    @patch('common.storage.tempfile')
    @patch('common.storage.os')
    def test_fsc_get_file_with_retry_fail(self, mock_os, mock_tempfile, mock_time):
        with patch.multiple(self.file_local, get=DEFAULT) as mocks:
            fsc = FileStorageClient()
            mock_tempfile.NamedTemporaryFile.return_value.name = "local_filename"
            mocks['get'].return_value = False
            self.assertIsNone(fsc._get_fileobj_with_retry(self.file_local))

            mock_os.remove.side_effect = OSError
            self.assertIsNone(fsc._get_fileobj_with_retry(self.file_local))

if __name__ == '__main__':
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    logger.disabled = True
    unittest.main()

