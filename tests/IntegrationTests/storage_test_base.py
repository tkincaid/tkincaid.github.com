import os
import shutil
import time
import unittest
from bson import ObjectId

from mock import patch
from config.engine import EngConfig

from common.storage import FileObject
from common.engine import metrics

class StorageTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pid = ObjectId('5373b2a98bd88f655d884aee')
        tests_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.test_directory = os.path.join(tests_dir,'testworkspacetemp')
        self.testdatadir = os.path.join(tests_dir,'testdata')
        self.patchers = []

        file_storage_patcher = patch.dict(EngConfig, {
            'LXC_CONTEXT_BASE': self.test_directory,
            'LOCAL_FILE_STORAGE_DIR' : os.path.join(self.test_directory, 'local_file_storage'),
            'CACHE_DIR' : os.path.join(self.test_directory,'cache'),
            # 'DATA_DIR' : os.path.join(self.test_directory,'temp')
            }, clear = False)

        file_storage_patcher.start()
        self.patchers.append(file_storage_patcher)

        self.clear_test_dir()
        os.mkdir(self.test_directory)
        # os.mkdir(EngConfig['DATA_DIR'])

    @classmethod
    def tearDownClass(self):
        self.clear_test_dir()
        self.stopPatching()

    @classmethod
    def stopPatching(self):
        for patcher in self.patchers:
            if patcher:
                patcher.stop()

    @classmethod
    def create_test_files(self, datasets = None):
        if not datasets:
            datasets = self.get_default()

        self.clear_test_dir()

        os.mkdir(self.test_directory)

        for each in datasets:
            testdatafile = each['filename'] if isinstance(each, dict) else each
            fin = os.path.join(self.testdatadir, testdatafile)
            fout = os.path.join(self.test_directory, testdatafile)
            shutil.copy(fin,fout)
            testdatafile = 'projects/'+str(self.pid)+'/raw/' + testdatafile
            FileObject(testdatafile).put(fin)

        return self.test_directory, datasets

    @classmethod
    def get_default(self):
        datasets = []
        datasets.append({'filename':'credit-sample-200.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
        datasets.append({'filename':'allstate-nonzero-200.csv','target':['Claim_Amount','Regression'], 'metric': metrics.GINI_NORM})
        datasets.append({'filename':'kickcars-sample-200.csv','target':['IsBadBuy','Binary'], 'metric': metrics.LOGLOSS})
        datasets.append({'filename':'credit-test-200.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
        datasets.append({'filename':'credit-test-NA-200.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
        datasets.append({'filename':'credit-train-small.csv','target':[u'SeriousDlqin2yrs','Binary'], 'metric': metrics.LOGLOSS})
        datasets.append({'filename':'fastiron-sample-400.csv','target':[u'SalePrice','Regression'], 'metric': metrics.GAMMA_DEVIANCE})
        datasets.append({'filename':'download_test.csv','target':[u'y','Regression'], 'metric': metrics.MAD})
        datasets.append({'filename':'download_test.csv','target':[u'c','Classification'], 'metric': metrics.LOGLOSS})
        return datasets

    @classmethod
    def clear_test_dir(self):
        attempts = 0
        while os.path.isdir(self.test_directory):
            attempts +=1
            if attempts == 3:
                raise Exception ('Could not remove dir: {}'.format(self.test_directory))
            shutil.rmtree(self.test_directory)
            time.sleep(0.1)
