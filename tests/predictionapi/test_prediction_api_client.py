import unittest

from profiling.apitests.prediction_client import read_input_file

class ReadFileTestCase(unittest.TestCase):

    def test_read_whole_file(self):
        filename = 'tests/testdata/credit-sample-200.csv'

        #no arguments, just read the whole file
        out = read_input_file(filename)
        for n, data in enumerate(out):
            pass
        self.assertEqual(n,0)
        self.assertEqual(len(data.split('\n')), 202)

    def test_read_only_10_rows(self):
        filename = 'tests/testdata/credit-sample-200.csv'

        #read only 10 rows
        out = read_input_file(filename,rows=10)
        for n, data in enumerate(out):
            pass
        self.assertEqual(n,0)
        self.assertEqual(len(data.split('\n')), 12)

    def test_read_10rows_batchesof2(self):
        filename = 'tests/testdata/credit-sample-200.csv'

        #read 10 rows, in batches of 2
        out = read_input_file(filename,rows=10,batchsize=2)
        for n, data in enumerate(out):
            self.assertEqual(len(data.split('\n')), 4)

        self.assertEqual(n,4)

    def test_read_500rows_load_test(self):
        filename = 'tests/testdata/credit-sample-200.csv'

        out = read_input_file(filename, rows=500, load_test=True)
        for n, data in enumerate(out):
            pass
        self.assertEqual(n,0)
        self.assertEqual(len(data.split('\n')), 501)

    def test_read_500rows_batchsize100_load_test(self):
        filename = 'tests/testdata/credit-sample-200.csv'

        out = read_input_file(filename, rows=500, batchsize=100, load_test=True)
        for n, data in enumerate(out):
            print n, len(data.split('\n'))
            self.assertEqual(len(data.split('\n')), 102)

        self.assertEqual(n,4)



if __name__ == '__main__':
    unittest.main()


