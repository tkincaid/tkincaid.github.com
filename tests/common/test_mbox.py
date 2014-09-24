import uuid

import unittest
from mock import Mock, patch
import pytest

import common.services.message_box as mbox


class TestMessageBoxConstructors(unittest.TestCase):

    def setUp(self):
        self.box_id = 'obscure-box-id-number'

    def test_init_with_a_dummy(self):
        tempstore = Mock()
        test = mbox.MessageBox(tempstore, self.box_id)

    @patch('common.wrappers.database', autospec=True)
    def test_default_init(self, fake_db):
        test = mbox.MessageBox.default_connection(self.box_id)


class TestMessageBoxFunctionality(unittest.TestCase):

    def setUp(self):
        self.box_id = str(uuid.uuid4())
        self.box = mbox.MessageBox.default_connection(self.box_id)

    @pytest.mark.db
    def test_read_before_write_results_in_None(self):
        result = self.box.read(timeout=0)
        self.assertIsNone(result)

    def test_box_id_None_returns_None(self):
        self.box = mbox.MessageBox.default_connection(None)
        self.box.write('Hello World')
        self.assertIsNone(self.box.read())

    @pytest.mark.db
    def test_second_read_is_empty(self):
        self.box.write('Hello World')
        read1 = self.box.read()
        read2 = self.box.read(timeout=0)
        self.assertEqual(read1, 'Hello World')
        self.assertIsNone(read2)







