import unittest
import datetime

import pytest
import numpy as np
from bson.objectid import ObjectId

from common.utilities import json_utils

class TestJSONUtils(unittest.TestCase):

    @pytest.mark.unit
    def test_json_transform(self):
        result = json_utils.json_transform({"key": 23})
        self.assertEqual(result, {"key": 23})

        result = json_utils.json_transform({23: "value"})
        self.assertEqual(result, {23: "value"})

        result = json_utils.json_transform({"key": np.nan})
        self.assertEqual(result, {"key": ''})

        result = json_utils.json_transform({"key": float("nan")})
        self.assertEqual(result, {"key": ''})

        result = json_utils.json_transform({"key": np.inf})
        self.assertEqual(result, {"key": ''})

        result = json_utils.json_transform({"key": float("inf")})
        self.assertEqual(result, {"key": ''})

        result = json_utils.json_transform({"key": np.float64("nan")})
        self.assertEqual(result, {"key": ''})

        _id = ObjectId('313233343536373839303930')
        result = json_utils.json_transform(_id)
        self.assertEqual(result, '313233343536373839303930')

        test_date = datetime.datetime(2013, 4, 1, 12, 23, 17)
        result = json_utils.json_transform(test_date)
        self.assertEqual(result, "2013-04-01 12:23:17")
