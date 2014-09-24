############################################################################
#
#       unit test for utilities
#
#       Author: ??
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################

import datetime
import flask
import unittest
from bson.objectid import ObjectId
from flask import json, make_response
from mock import patch

from test_base import TestBase
from MMApp.utilities.web_response import *
import MMApp.app

import pytest

class UtilitiesTestCase(unittest.TestCase):

    def setUp(self):
        self.app = MMApp.app.app

    @pytest.mark.unit
    def test_make_json_response(self):
        with self.app.test_request_context('/'):
            response = make_json_response({'change': 'none'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'application/json')
            self.assertEqual('{"change": "none"}', response.data)

    @pytest.mark.unit
    def test_make_json_response_nans(self):
        """Make sure that we're raising a TypeError if any data that
        is not strict JSON makes it through our serializer"""
        with self.app.test_request_context('/'):
            with patch('common.utilities.json_utils.json_transform') as mock_json_transform:
                mock_json_transform.return_value = {"key": float("nan")}
                self.assertRaises(ValueError, make_json_response, {"key": float("nan")})

    @pytest.mark.unit
    def test_bad_id_response(self):
        with self.app.test_request_context('/'):
            bad_response = bad_id_response()
            self.assertEqual(bad_response.status_code, 400)
            self.assertEqual(bad_response.mimetype, 'application/json')
            self.assertEqual(bad_response.data, '{"message": "invalid id"}')


if __name__ == '__main__':
    unittest.main()
