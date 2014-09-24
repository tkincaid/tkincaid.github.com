import unittest

import numpy as np


class MockUserModule(object):
    '''A mock user module to use for testing'''
    class CustomModel(object):
        def fit(self, x, y):
            return self
        def predict(self,x):
            return np.ones((len(x),1))

class UserModelTestBase(unittest.TestCase):

    @classmethod
    def generate_one_request(cls):
        return {
            "_id": "5241b1ba637aba4ed9896420",
            "pid": "5241b1ba637aba4ed9896420",
            "uid": "5241b1ba637aba4ebf896422",
            "lid": "new",
            "blueprint": {},
            "bp": 1,
            "project": {
                "target": {"type": "Binary", "name": "y", "size": 4500.0},
                "default_dataset_id": "958d0c63-1cb5-4932-acf9-915251945adc",
                "partition": {"folds": 5, "reps": 5},
                "active": 1,
                'holdout_pct': 20,
                "_id": "5241c377637aba69b9809faf",
                "uid": "5241c377637aba69a1809fb1",
                "stage": "modeling",
                "metadata": {"created": "2013-09-24 16:53:11.617000", "pid": "5241c377637aba69b9809faf", "varTypeString": "NNCNNCTCTCCNCNCCCNNNNNNNNCCNNCNNNNC", "shape": [5000, 35], "dataset_id": "958d0c63-1cb5-4932-acf9-915251945adc", "originalName": "/home/ulises/workspace/DataRobot/tests/testdata/kickcars.rawdata.csv", "_id": "5241c3775ec912fd9cf3b1f4", "columns": ["RefId", "PurchDate", "Auction", "VehYear", "VehicleAge", "Make", "Model", "Trim", "SubModel", "Color", "Transmission", "WheelTypeID", "WheelType", "VehOdo", "Nationality", "Size", "TopThreeAmericanName", "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice", "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1", "VNST", "VehBCost", "IsOnlineSale", "WarrantyCost", "y", "split"], "files":['FAKE']},
                },
            "filename": "/home/ulises/workspace/DataRobot/tests/testdata/kickcars.rawdata.csv",
            "shape": [5000, 35],
            "active": 1,
            "columns": ["RefId", "PurchDate", "Auction", "VehYear", "VehicleAge", "Make", "Model", "Trim", "SubModel", "Color", "Transmission", "WheelTypeID", "WheelType", "VehOdo", "Nationality", "Size", "TopThreeAmericanName", "MMRAcquisitionAuctionAveragePrice", "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice", "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice", "MMRCurrentRetailCleanPrice", "PRIMEUNIT", "AUCGUART", "BYRNO", "VNZIP1", "VNST", "VehBCost", "IsOnlineSale", "WarrantyCost", "y", "split"],
            "stage": "modeling",
            "icons": [4],
            "qid": "24",
            "model_type": "user model 1",
            "modelpredict": "function(model,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  predict.gbm(model,datasub,n.trees=500,type=\"response\");\n}\n",
            "modelfit": "function(response,data) {\n  library(gbm);\n  datasub = data[,c(\"VehYear\",\"VehBCost\")];\n  gbm.fit(datasub,response,n.trees=500, interaction.depth=10,shrinkage=0.1,bag.fraction=0.5,keep.data=FALSE, verbose=FALSE);\n}\n",
            "user_files": {},
            "max_reps": 1,
            "user_item": 1,
            "samplesize": 1000,
            "command": "fit",
            "key": "1",
            "max_folds": 0,
            "features": "",
            'partitions': [[0]] # <-- Not real
            }

    @classmethod
    def generate_one_py_request(cls):
        r = cls.generate_one_request()
        del r['modelfit']
        del r['modelpredict']
        r['modelsource'] = str(
                'import numpy\n'
                'import pandas\n'
                'class CustomModel(object):\n'
                '    def fit(self, X,Y):\n'
                '        return self\n'
                '\n'
                '    def predict(self, X):\n'
                '        return pd.series(np.ones(len(X)))\n')
        r['classname'] = 'CustomModel'
        return r


