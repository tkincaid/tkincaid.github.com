import os
import json

import unittest
import pytest
from mock import patch
from cStringIO import StringIO

import numpy as np
import pandas as pd

import common.io.disk_access as da
import common.exceptions as ex


TESTDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '../',
                            'testdata')
IO_EXAMPLES_DIR = os.path.join(TESTDATA_DIR, 'io')


class TestFilenameInspection(unittest.TestCase):

    def test_no_extension_assumed_csv(self):
        filename = 'foobar'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, filename)
        self.assertIsNone(fname.extension)
        self.assertIsNone(fname.compression)
        self.assertIsNone(fname.archival)

    def test_csv_extension_assumed_csv(self):
        filename = 'foobar.csv'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertIsNone(fname.compression)
        self.assertIsNone(fname.archival)

    def test_xls_extensions_assumed_xls(self):
        filename = 'foobar.xls'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'xls')
        self.assertIsNone(fname.compression)
        self.assertIsNone(fname.archival)

    def test_xlsx_extensions_assumed_xls(self):
        filename = 'foobar.xlsx'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'xlsx')
        self.assertIsNone(fname.compression)
        self.assertIsNone(fname.archival)

    def test_csv_gz_extensions_assumed_csv(self):
        filename = 'foobar.csv.gz'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertEqual(fname.compression, 'gz')
        self.assertIsNone(fname.archival)

    def test_csv_bz2_extensions_assumed_csv(self):
        filename = 'foobar.csv.bz2'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertEqual(fname.compression, 'bz2')
        self.assertIsNone(fname.archival)

    def test_caps_okay_for_extension_and_compression(self):
        filename = 'foobar.CSV.GZ'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertEqual(fname.compression, 'gz')
        self.assertIsNone(fname.archival)

    def test_compressed_excel_unacceptable(self):
        filename = 'foobar.xls.gz'
        with self.assertRaises(ex.InvalidFilenameError):
            da.FileExtensionClues(filename)

    def test_tar_gz_okay(self):
        filename = 'foobar.csv.tar.gz'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertEqual(fname.compression, 'gz')
        self.assertEqual(fname.archival, 'tar')

    def test_tgz_okay(self):
        filename = 'foobar.csv.tgz'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertIsNone(fname.compression)
        self.assertEqual(fname.archival, 'tgz')

    def test_tar_bz2_okay(self):
        filename = 'foobar.csv.tar.bz2'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertEqual(fname.compression, 'bz2')
        self.assertEqual(fname.archival, 'tar')

    def test_zip_csv_okay(self):
        filename = 'foobar.csv.zip'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertEqual(fname.extension, 'csv')
        self.assertIsNone(fname.compression)
        self.assertEqual(fname.archival, 'zip')

    def test_zip_no_ext_okay(self):
        filename = 'foobar.zip'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'foobar')
        self.assertIsNone(fname.extension)
        self.assertIsNone(fname.compression)
        self.assertEqual(fname.archival, 'zip')

    def test_arbitrary_number_of_prefixes_okay(self):
        filename = 'kickcars.chas.a-eda.chewbacca.csv'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'kickcars')
        self.assertEqual(fname.extension, 'csv')
        self.assertIsNone(fname.compression)
        self.assertIsNone(fname.archival)

    def test_zip_with_arbitrary_prefixes_okay(self):
        filename = 'kickcars.chas.a-eda.chewbacca.zip'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'kickcars')
        self.assertIsNone(fname.extension)
        self.assertIsNone(fname.compression)
        self.assertEqual(fname.archival, 'zip')

    def test_gz_with_arbitrary_prefixes_okay(self):
        filename = 'kickcars.chas.a-eda.chewbacca.gz'
        fname = da.FileExtensionClues(filename)
        self.assertEqual(fname.name, 'kickcars')
        self.assertIsNone(fname.extension)
        self.assertEqual(fname.compression, 'gz')
        self.assertIsNone(fname.archival)

    def test_nonsense_extension_unacceptable(self):
        filename = 'foobar.grue'
        with self.assertRaises(ex.InvalidFilenameError):
            da.FileExtensionClues(filename)


@pytest.mark.integration
class TestFileInspections(unittest.TestCase):

    @patch('common.io.disk_access.os.remove')
    def test_inspect_csv(self, fake_remove):
        filename = 'bcwisconsin.csv'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'delimiter': ',',
            'type': 'csv',
            'encoding': 'ASCII',
            'archival': None,
            'compression': None
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_csv_bz2(self, fake_remove):
        filename = 'bcwisconsin.csv.bz2'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'delimiter': ',',
            'type': 'csv',
            'encoding': 'ASCII',
            'archival': None,
            'compression': 'bz2'
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_csv_gz(self, fake_remove):
        filename = 'bcwisconsin.csv.gz'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'delimiter': ',',
            'type': 'csv',
            'encoding': 'ASCII',
            'archival': None,
            'compression': 'gz'
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_csv_tgz(self, fake_remove):
        filename = 'bcwisconsin.tgz'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'delimiter': ',',
            'type': 'csv',
            'archived_name': 'bcwisconsin.csv',
            'encoding': 'ASCII',
            'archival': 'tgz',
            'compression': None
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_zip(self, fake_remove):
        filename = 'bcwisconsin.zip'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'delimiter': ',',
            'type': 'csv',
            'encoding': 'ASCII',
            'archival': 'zip',
            'archived_name': 'bcwisconsin.csv',
            'compression': None
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_xlsx(self, fake_remove):
        filename = 'kickcars.xlsx'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'type': 'xls',
            'sheet': 'Sheet1',
            'archival': None,
            'compression': None
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_zipped_excel(self, fake_remove):
        filename = 'kickcars_excel.zip'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'type': 'xls',
            'sheet': 'Sheet1',
            'archival': 'zip',
            'archived_name': 'kickcars.xlsx',
            'compression': None
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_tgz_excel(self, fake_remove):
        filename = 'kickcars_excel.tgz'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'type': 'xls',
            'sheet': 'Sheet1',
            'archival': 'tgz',
            'archived_name': 'kickcars.xlsx',
            'compression': None
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_numeric_column_excel(self, fake_remove):
        filename = 'numeric_colname.xlsx'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            'type': 'xls',
            'sheet': 'Sheet1',
            'archival': None,
            'compression': None
        }
        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_inspect_unicode_names(self, fake_remove):
        filename = 'international.csv'
        filepath = os.path.join(IO_EXAMPLES_DIR, filename)
        df, control = da.inspect_uploaded_file(filepath, filename)

        reference_control = {
            u'type': u'csv',
            'archival': None,
            'compression': None,
            u'encoding': u'UTF-8',
            u'delimiter': u','
        }

        self.assertEqual(control, reference_control)
        self.assertFalse(fake_remove.called)


@pytest.mark.integration
class TestFileInspectionsNegatives(unittest.TestCase):

    IO_REJECTS_DIR = os.path.join(IO_EXAMPLES_DIR, 'reject')

    @patch('common.io.disk_access.os.remove')
    def test_reject_compression_within_archive(self, fake_remove):
        filename = 'bad-nested.tgz'
        filepath = os.path.join(self.IO_REJECTS_DIR, filename)
        with self.assertRaises(ex.NestedCompressionError):
            da.inspect_uploaded_file(filepath, filename)
            self.assertTrue(fake_remove.called)

    @patch('common.io.disk_access.os.remove')
    def test_reject_multiple_files_in_archive(self, fake_remove):
        filename = 'bad-multiple-files.tgz'
        filepath = os.path.join(self.IO_REJECTS_DIR, filename)
        with self.assertRaises(ex.MultipleCompressedFilesError):
            da.inspect_uploaded_file(filepath, filename)
            self.assertTrue(fake_remove.called)


@pytest.mark.unit
class TestFileInspectionErrorConditions(unittest.TestCase):

    def test_reject_upon_too_large_file_upload(self):
        fhandle = StringIO()
        fhandle.write('TEN CHARS.')
        fhandle.seek(0)
        with self.assertRaises(ex.IncompleteFileReadError):
            da.paranoid_read(fhandle, 5)

    def test_unarchive_with_unknown_archival_type(self):
        fpath = 'irrelevant.csv.mattress'
        bad_archival_type = 'mattress'

        with self.assertRaises(ex.InvalidFilenameError):
            da.get_unarchived_file(fpath, bad_archival_type, 'blah.csv')


class TestReadJson(unittest.TestCase):

    def test_constructs_a_usable_dataframe(self):
        data = [{'a': 1, 'b': 2, 'c': 3},
                {'a': 4, 'b': 5, 'c': 6}]
        enc = json.dumps(data)
        frame = da.read_json(enc)
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(frame.shape, (2, 3))
        self.assertTrue(np.all(frame.columns == ['a', 'b', 'c']))

    def test_differing_column_number_rejected(self):
        data = [{'a': 1, 'b': 2, 'c': 3},
                {'a': 4, 'b': 5, 'c': 6, 'd': 7}]
        enc = json.dumps(data)
        with self.assertRaisesRegexp(
                ex.JSONLayoutError, 'Record number \d has \d fields'):
            da.read_json(enc)

    def test_single_record_rejected(self):
        data = {'a': 1, 'b': 2, 'c': 3}
        enc = json.dumps(data)
        with self.assertRaisesRegexp(ex.JSONLayoutError,
                                     'formatted as an array of objects'):
            da.read_json(enc)

    def test_dict_of_lists_rejected(self):
        data = {'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}
        enc = json.dumps(data)
        with self.assertRaisesRegexp(ex.JSONLayoutError,
                                     'formatted as an array of objects'):
            da.read_json(enc)

    def test_empty_json_rejected(self):
        with self.assertRaisesRegexp(ex.JSONLayoutError,
                                    'empty JSON'):
            da.read_json('[]')
