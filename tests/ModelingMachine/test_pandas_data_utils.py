############################################################################
#
#       unit test pandas data wrangling utils
#
#       Author: Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
###########################################################################
import pandas as pd
import numpy as np
import unittest

from datetime import datetime

from ModelingMachine.engine import pandas_data_utils as pdu


class PandasDataUtilsTest(unittest.TestCase):

    def test_is_date_smoke(self):
        """Smoke test. """
        s = pd.Series(['1/1/2013', '2/1/2013'])
        fmt = pdu.isDate(s)
        self.assertEqual(fmt, '%m/%d/%Y')

    def test_is_date_formats(self):
        test = ['%m/%d/%Y', '%m/%d/%y', '%m-%d-%Y', '%m-%d-%y',
                '%Y/%m/%d', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
                '%m/%d/%Y %H:%M',
                '%m/%d/%Y %H:%M:%S', '%m/%d/%y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
                '%m-%d-%y %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%S.%f',
                '%m %d %Y %H %M %S', '%m %d %y %H %M %S',
                '%M:%S', '%H:%M', '%H:%M:%S',
                '%Y %m %d %H %M %S', '%Y %m %d']
        dt = datetime(1970, 1, 21, 13, 59)
        for fmt in test:
            s = pd.Series([dt.strftime(fmt)] * 100)
            guessed_fmt = pdu.isDate(s)
            self.assertEqual(fmt, guessed_fmt)

    def test_is_date_iso(self):
        """Test iso format"""
        s = pd.Series(map(datetime.fromordinal, np.random.randint(139338, 200000, size=100)))
        iso_s = s.map(lambda x: x.isoformat())
        fmt = pdu.isDate(iso_s)
        self.assertEqual(fmt, '%Y-%m-%dT%H:%M:%S')

        iso_s = pd.Series([datetime.utcnow().isoformat() for i in range(100)])
        fmt = pdu.isDate(iso_s)
        self.assertEqual(fmt, '%Y-%m-%dT%H:%M:%S.%f')

        # Now with Z at the end
        iso_s_z = iso_s.map(lambda x: x + 'Z')
        fmt = pdu.isDate(iso_s_z)
        self.assertEqual(fmt, '%Y-%m-%dT%H:%M:%S.%fZ')

    def test_is_date_questionmark(self):
        """Regression test - isDate mistakes allstate OrdCat for date (not always)"""
        rs = np.random.RandomState(13)
        x = rs.randint(2, 6, size=200)
        x = map(str, x)
        s = pd.Series(x)
        s[3] = '?'

        for i in range(10):
            # if the sample includes ? it will give datetime
            fmt = pdu.isDate(s)
            self.assertEqual(fmt, '')

    def test_is_length(self):
        test_data = pd.DataFrame({
            'a': ['4\' 5"',' 5"','4\'','4"','UNSPECIFIED',float('nan')],
            'b': ['4\' a','x','x','x','x','x'],
            'c': ['4\' a','x','x','x','x','x'],
            'd': ['4` 5"','x','x','x','x','x'],
            'e': ['4 "','x',float('nan'),'x','x','x']
        })
        self.assertTrue(pdu.isLength(test_data['a']))
        self.assertFalse(pdu.isLength(test_data['b']))
        self.assertFalse(pdu.isLength(test_data['c']))
        self.assertFalse(pdu.isLength(test_data['d']))
        self.assertFalse(pdu.isLength(test_data['e']))
 
    def test_iscurrency_ranges_if_false(self):
        """Test that is currency identifies ranges as text."""
        inp = pd.DataFrame({'a': [ "$500;000-$774;999", "$500;000-$774;999",
                "$300;000-$349;999", "$300;000-$349;999",  "$300;000-$349;999",
                "$300;000-$349;999",  "$450;000-$499;999", "$250;000-$274;999",
                "$175;000-$199;999",  "$175;000-$199;999"],
            'b': [str(i) for i in xrange(10)] })
        X = pdu.isCurrency(inp['a'])
        self.assertFalse(X)

    def test_currency_parser(self):
        """Test that is currency identifies ranges as text."""
        data = pd.Series(np.array(["$30", "$4,500", "$45.10", "$340,738,213.49"]))
        assert list(pdu.currencyParser(data)) == [30.0, 4500.0, 45.1, 340738213.49]

    def test_is_text_lorem_ipsum_is_text(self):
        """Test that is_text correctly identifies text
        """
        test_data = pd.DataFrame({
            'a': ["Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor",
                  "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud",
                  "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute",
                  "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla",
                  "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia",
                  "deserunt mollit anim id est laborum."]
        })
        self.assertTrue(pdu.isText(test_data['a']))

    def test_is_text_categories_with_spaces_not_text(self):
        """Test that is_text does not classify categorical data with spaces as text
        """
        test_data = pd.DataFrame({
            'a': ["None or Unspecified", "Wheel Loader",  "Hydraulic Excavator, Track", "Skid Steer Loaders",
                "Motorgrader - 45.0 to 130.0 Horsepower", "Track Type Tractor, Dozer", "Four Wheel Drive"]
            })
        self.assertTrue(pdu.isText(test_data['a']) is None)

    def test_is_text_categories_dates_with_times(self):
        """Test that is_text does not classify dates with times as text
        """
        test_data = pd.DataFrame({
            'a': ["2012-01-01 13:59:41", "2012-01-01 01:20:08",  "2012-01-02 20:29:54", "2012-01-02 19:04:51",
                "2012-02-03 15:37:27", "2012-01-03 17:14:10", "2012-01-03 18:26:22"]
            })
        
        self.assertTrue(pdu.isText(test_data['a']) is None)

    def test_is_text_nones(self):
        """Test that is_text does not crash when given a column of Nones
        """
        test_data = pd.DataFrame({
            'a': [None, None,  None, None, None, None, None]
            })
        self.assertTrue(pdu.isText(test_data['a']) is None)

if __name__ == '__main__':
    unittest.main()
