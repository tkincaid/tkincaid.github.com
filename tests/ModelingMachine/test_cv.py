import unittest
import sys
import os
import numpy as np
import pandas
import multiprocessing
import inspect
import hmac
from itertools import product
from ModelingMachine.engine.new_partition import NewPartition
from ModelingMachine.engine.cross_validation import CVbyKeyColdStart,DateCV

def get_Z(r):
    # helper function for parallel RNG test
    return NewPartition(20,cv_method='RandomCV',seed=r)

class TestCVClasses(unittest.TestCase):
    def test_samplepct(self):
        # make sure larger sample % contains all of the rows from smaller sample %
        Z = NewPartition(40,cv_method='RandomCV',reps=5,total_size=45,seed=0)
        Z.set(max_folds=0,samplepct=50)
        train50 = Z.T(r=0,k=-1)
        Z = NewPartition(40,cv_method='RandomCV',reps=5,total_size=45,seed=0)
        Z.set(max_folds=0,samplepct=64)
        train64 = Z.T(r=0,k=-1)
        self.assertTrue(set(train50) < set(train64))
        Z = NewPartition(40,cv_method='RandomCV',reps=1,total_size=45,seed=0)
        Z.set(max_folds=0,samplepct=65)
        train65 = Z.T(r=-1,k=-1)
        Z = NewPartition(40,cv_method='RandomCV',reps=1,total_size=45,seed=0)
        Z.set(max_folds=0,samplepct=82)
        train82 = Z.T(r=-1,k=-1)
        self.assertTrue(set(train65) < set(train82))

    def test_new_partition(self):
        # test old style methods
        # Check that samplepct works
        Z = NewPartition(20,cv_method='RandomCV',samplepct=64,reps=5,total_size=25,seed=0)
        Z.set(max_folds=0)
        # with 5-folds and a 20% (1-20/25) holdout, 64% should use all non-holdout data
        self.assertEqual(Z.samplesize,20)
        train_sizes = (16,16,16,16,16)
        folds = 0
        for p in Z:
            train = Z.T(**p)
            test = Z.S(**p)
            self.assertEqual(len(set(train)),train_sizes[folds])
            self.assertEqual(len(test),Z.size-train_sizes[folds])
            folds += 1
        # check iterator loops correct number of times
        self.assertEqual(folds,5)
        # check A() method returns all rows
        self.assertEqual(Z.A(),range(Z.size))
        # check A() method returns no dups
        self.assertEqual(len(set(Z.A())),Z.size)

        # test sample size < 64
        Z.set(samplepct=50,max_folds=0)
        # test set size = non-holdout-size * 1/folds
        test_size = int(Z.size * 1.0/Z.n_reps)
        self.assertEqual(test_size,4)
        # samplesize = test_size + train_size
        self.assertEqual(Z.samplesize,int(0.50*Z.total_size+test_size))
        train_sizes = (12,13,13,13,13)
        folds = 0
        for p in Z:
            train = Z.T(**p)
            test = Z.S(**p)
            self.assertEqual(len(set(train)),train_sizes[folds])
            self.assertEqual(len(test),test_size)
            folds += 1
        # check iterator loops correct number of times
        self.assertEqual(folds,5)
        # check A() method returns correct number of rows
        self.assertEqual(len(Z.A()),Z.samplesize)
        # check A() method returns no dups
        self.assertEqual(len(set(Z.A())),Z.samplesize)

        # test 64 < sample size <=80
        Z = NewPartition(20,cv_method='RandomCV',reps=1,total_size=20,seed=0)
        Z.set(max_folds=0,samplepct=70,partitions=[[-1,-1]])
        # no test set, so samplesize = train_size
        self.assertEqual(Z.samplesize,int(0.70*Z.total_size))
        train_sizes = (14,)
        folds = 0
        for p in Z:
            train = Z.T(**p)
            test = Z.S(**p)
            self.assertEqual(len(set(train)),train_sizes[folds])
            self.assertEqual(0,len(test))
            folds += 1
        # check iterator loops correct number of times
        self.assertEqual(folds,1)
        # check A() method returns correct number of rows
        self.assertEqual(len(Z.A()),Z.samplesize)
        # check A() method returns no dups
        self.assertEqual(len(set(Z.A())),Z.samplesize)

        # test sample size > 80
        Z = NewPartition(20,cv_method='RandomCV',reps=1,total_size=20,seed=0)
        Z.set(max_folds=0,samplepct=90,partitions=[[-1,-1]])
        # no test set, so samplesize = train_size
        self.assertEqual(Z.samplesize,int(0.90*Z.total_size))
        train_sizes = (18,)
        folds = 0
        for p in Z:
            train = Z.T(**p)
            test = Z.S(**p)
            self.assertEqual(len(set(train)),train_sizes[folds])
            self.assertEqual(0,len(test))
            folds += 1
        # check iterator loops correct number of times
        self.assertEqual(folds,1)
        # check A() method returns correct number of rows
        self.assertEqual(len(Z.A()),Z.samplesize)
        # check A() method returns no dups
        self.assertEqual(len(set(Z.A())),Z.samplesize)

    def test_RandomCV_CV(self):
        # test RandomCV class with folds=5
        # Some test data
        test_data = pandas.DataFrame({
            'key1':['a','a','a','b','b','b','c','c','c','d','d','e','f','g','h','h'],
            'data':range(16),
            'y':range(16)
        })
        Z = NewPartition(16,cv_method='RandomCV')
        yt = test_data['y']
        # create RandomCV class with 5 fold
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=0)
        self.assertEqual(cv.__class__.__name__,'RandomCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # Partition sizes for seed 0.  These should be correct
        # unless the RandomCV code is changed (which it should not be).
        train_sizes = (12,13,13,13,13)
        test_rows = []
        for k in range(5):
            (train,test) = cv[k]
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # check partition sizes
            self.assertEqual(len(set(train)),train_sizes[k])
            self.assertEqual(len(test),16-train_sizes[k])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
        # same thing using cv as an iterator
        test_rows = []
        k = 0
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(set(train)),train_sizes[k])
            self.assertEqual(len(test),16-train_sizes[k])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            k += 1
        # check iterator does the correct numeber of iterations
        self.assertEqual(k,5)
        # make sure each row was used for test set
        self.assertEqual(len(test_rows),16)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),16)
        # make sure data was shuffled
        self.assertTrue(any([test_rows[i]>test_rows[i+1] for i in range(len(test_rows)-1)]))
        # test all() method
        self.assertEqual(cv.all().tolist(),range(16))
        # test len() method
        self.assertEqual(len(cv),5)
        test_sets1 = [fold[1] for fold in cv]
        # test new random seed produces different results
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=1)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertNotEqual(test_rows,test_rows2)
        test_sets2 = [fold[1] for fold in cv]
        for i in range(k):
            # Ensure at least half of each fold is differnt
            num_in_common = len(np.intersect1d(test_sets1[i], test_sets2[i]))
            self.assertLessEqual(num_in_common, len(test_sets1[i]) / 2)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)
        test_sets2 = [fold[1] for fold in cv]
        for i in range(k):
            # Ensure each fold is the same
            num_in_common = len(np.intersect1d(test_sets1[i], test_sets2[i]))
            self.assertEqual(num_in_common, len(test_sets1[i]))

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=0,samplesize=12)
        test_rows = []
        train_sizes_s12 = (9,9,10,10,10)
        for k in range(5):
            (train,test) = cv[k]
            # Check partition sizes
            self.assertEqual(len(train),train_sizes_s12[k])
            # sample size shouldn't change test size
            self.assertEqual(len(test),16-train_sizes[k])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

    def test_RandomCV_testfrac(self):
        # test RandomCV class with test_frac=0.25
        # Some test data
        test_data = pandas.DataFrame({
            'key1':['a','a','a','b','b','b','c','c','c','d','d','e','f','g','h','h'],
            'data':range(16),
            'y':range(16)
        })
        Z = NewPartition(16,cv_method='RandomCV')
        yt = test_data['y']
        # create RandomCV class with 25% test set
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=.25,random_state=0)
        self.assertEqual(cv.__class__.__name__,'RandomCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        (train,test) = cv[0]
        # Check partition sizes
        self.assertEqual(len(train),12)
        self.assertEqual(len(test),4)
        # check that rows don't appear in both train & test
        self.assertTrue(set(train) & set(test) == set())
        # same thing using cv as an iterator
        test_rows = []
        k = 0
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # Check partition sizes
            self.assertEqual(len(train),12)
            self.assertEqual(len(test),4)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            k += 1
        # check iterator does the correct numeber of iterations
        self.assertEqual(k,1)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),4)
        # test all() method
        self.assertEqual(cv.all().tolist(),range(16))
        # test len() method
        self.assertEqual(len(cv),1)
        # test new random seed produces different results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=.25,random_state=1)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertNotEqual(test_rows,test_rows2)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=.25,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=.25,random_state=0,samplesize=12)
        test_rows = []
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(train),8)
            # sample size shouldn't change test size
            self.assertEqual(len(test),4)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

    def test_CVbyKey_testfrac(self):
        # test CV by keys train/test split
        test_data = pandas.DataFrame({
            'key1':['a','a','a','b','b','b','c','c','c','d','d','e','f','g','h','h'],
            'data':range(16),
            'y':range(16)
        })
        # create CVbyKey class with 25% test set
        Z = NewPartition(16,cv_info=test_data['key1'],cv_method='GroupCV')
        yt = test_data['y']
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=0)
        self.assertEqual(cv.__class__.__name__,'CVbyKeyColdStart')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # check results
        folds = 0
        test_rows = []
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # train = 75%, test = 25%
            self.assertEqual(len(train),12)
            self.assertEqual(len(test),4)
            # big chunks should not be in the test set
            self.assertTrue(np.all(np.array(test) > 8))
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        # make sure cv only generates one train/test set
        self.assertEqual(folds,1)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),4)
        # test all() method
        self.assertEqual(cv.all().tolist(),range(16))
        # test len() method
        self.assertEqual(len(cv),1)
        # test new random seed produces different results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=2)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertNotEqual(test_rows,test_rows2)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=.25,random_state=0,samplesize=12)
        test_rows = []
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(train),9)
            # sample size shouldn't change test size
            self.assertEqual(len(test),4)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)


    def test_CVbyKey_CV(self):
        # test CV by keys 3 fold CV
        test_data = pandas.DataFrame({
            'key1':['a','a','a','a','a','a','a','a','b','b','b','b','b','b','b','b'],
            'key2':['a','a','b','b','c','c','d','d','d','d','e','e','f','g','h','i'],
            'data':range(16),
            'y':range(16)
        })
        # create CVbyKey class with 3 folds
        Z = NewPartition(16,cv_info=(test_data['key1']+test_data['key2']),cv_method='GroupCV')
        yt = test_data['y']
        cv = Z.get_cv(size=len(yt),yt=yt,folds=3,random_state=0)
        self.assertEqual(cv.__class__.__name__,'CVbyKeyColdStart')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # check results
        folds = 0
        train_sizes = (11,10,11)
        test_rows = []
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # check sizes are correct
            self.assertEqual(len(train),train_sizes[folds])
            self.assertEqual(len(test),16-train_sizes[folds])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        # make sure iterator does the correct number of iterations
        self.assertEqual(folds,3)
        # make sure each row was used for test set
        self.assertEqual(len(test_rows),16)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),16)
        # test all() method
        self.assertEqual(cv.all().tolist(),range(16))
        # test len() method
        self.assertEqual(len(cv),3)
        # test new random seed produces different results
        cv = Z.get_cv(size=len(yt),yt=yt,folds=3,random_state=2)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertNotEqual(test_rows,test_rows2)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,folds=3,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,folds=3,random_state=0,samplesize=12)
        test_rows = []
        train_sizes2 = (8,8,8)
        folds = 0
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(train),train_sizes2[folds])
            # sample size shouldn't change test size
            self.assertEqual(len(test),16-train_sizes[folds])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        # make sure iterator does the correct number of iterations
        self.assertEqual(folds,3)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

    def test_DateCV(self):
        # test CV by date with 25% test set
        np.random.seed(0)
        test_data = pandas.DataFrame({
            'date':np.random.permutation(16),
            'key2':['a','a','a','b','b','b','c','c','c','d','d','d','d','e','e','e'],
            'data':range(16),
            'y':range(16)
        })
        # create DateCV class with 25% test set
        Z = NewPartition(16,cv_info=test_data['date'],cv_method='DateCV',validation_pct=0.25)
        yt = test_data['y']
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=0)
        self.assertEqual(cv.__class__.__name__,'DateCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d13d12299d4981b9d61568c4fa6ccd3a")
        # check results
        folds = 0
        test_rows = []
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # check split sizes
            self.assertEqual(len(train),12)
            self.assertEqual(len(test),4)
            # make sure all train dates are less than all test dates
            date_prod = pandas.DataFrame(list(product(test_data['date'][train],test_data['date'][test])),columns=('train','test'))
            self.assertTrue(np.all(date_prod['train']<date_prod['test']))
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        self.assertEqual(folds,1)

        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),4)
        # test all() method
        self.assertEqual(cv.all().tolist(),range(16))
        # test len() method
        self.assertEqual(len(cv),1)
        # test new random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=1)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=.25,random_state=0,samplesize=12)
        test_rows = []
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(train),8)
            # sample size shouldn't change test size
            self.assertEqual(len(test),4)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

    def test_StratifiedCV(self):
        # test StratifiedCV class with folds=5
        # Some test data
        #TODO: handle the case where number of positve classes < folds
        test_data = pandas.DataFrame({
            'key1':['a','a','a','b','b','b','c','c','c','d','d','e','f','g','h','h'],
            'data':range(16),
            'y':[0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1]
        })
        yt = test_data['y']
        Z = NewPartition(16,yt=yt,cv_method='StratifiedCV')
        # create StratifiedCV class with 5 fold
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=0)
        self.assertEqual(cv.__class__.__name__,'StratifiedCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # Partition sizes for seed 0.  These should be correct
        # unless the RandomCV code is changed (which it should not be).
        train_sizes = (12,13,13,13,13)
        pos_count = (2,1,1,1,1)
        test_rows = []
        for k in range(5):
            (train,test) = cv[k]
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # check partition sizes
            self.assertEqual(len(set(train)),train_sizes[k])
            self.assertEqual(len(test),16-train_sizes[k])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            # check that positive classes are distributed correctly
            test_data['y'][test]
            self.assertEqual(test_data['y'][test].sum(),pos_count[k])
        # same thing using cv as an iterator
        test_rows = []
        k = 0
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(set(train)),train_sizes[k])
            self.assertEqual(len(test),16-train_sizes[k])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            k += 1
        # check iterator does the correct numeber of iterations
        self.assertEqual(k,5)
        # make sure each row was used for test set
        self.assertEqual(len(test_rows),16)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),16)
        # make sure data was shuffled
        self.assertTrue(any([test_rows[i]>test_rows[i+1] for i in range(len(test_rows)-1)]))
        # test all() method
        self.assertEqual(len(set(cv.all().tolist())),16)
        # test len() method
        self.assertEqual(len(cv),5)
        # test new random seed produces different results
        test_sets1 = [fold[1] for fold in cv]
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=1)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        test_sets2 = [fold[1] for fold in cv]
        for i in range(k):
            # Ensure at least half of each fold is differnt
            num_in_common = len(np.intersect1d(test_sets1[i], test_sets2[i]))
            self.assertLessEqual(num_in_common, len(test_sets1[i]) / 2)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,folds=5,random_state=0,samplesize=12)
        test_rows = []
        #TODO: better balance for small samples
        train_sizes_s12 = (9,9,10,10,10)
        for k in range(5):
            (train,test) = cv[k]
            # Check partition sizes
            self.assertEqual(len(train),train_sizes_s12[k])
            # sample size shouldn't change test size
            self.assertEqual(len(test),16-train_sizes[k])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

    def test_UserCV(self):
        # test UserCV by keys 3 fold CV
        test_data = pandas.DataFrame({
            'user_keys':['1','2','3','3','2','1','2','3','2','2','1','3','1','1','2','2'],
            'data':range(16),
            'y':range(16)
        })
        # create CVbyKey class with 3 folds
        Z = NewPartition(16,cv_info=test_data['user_keys'],cv_method='UserCV')
        yt = test_data['y']
        cv = Z.get_cv(size=len(yt),yt=yt,folds=3,random_state=0)
        self.assertEqual(cv.__class__.__name__,'UserCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # check results
        folds = 0
        train_sizes = (12,11,9)
        test_rows = []
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # check sizes are correct
            self.assertEqual(len(train),train_sizes[folds])
            self.assertEqual(len(test),16-train_sizes[folds])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        # make sure iterator does the correct number of iterations
        self.assertEqual(folds,3)
        # make sure each row was used for test set
        self.assertEqual(len(test_rows),16)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),16)
        # test all() method
        self.assertEqual(cv.all().tolist(),range(16))
        # test len() method
        self.assertEqual(len(cv),3)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,folds=3,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,folds=3,random_state=0,samplesize=12)
        test_rows = []
        train_sizes2 = (9,8,7)
        folds = 0
        for train,test in cv:
            # Check partition sizes
            #TODO: the sampling algorithm doesn't produce very balanced training sets
            self.assertEqual(len(train),train_sizes2[folds])
            # sample size shouldn't change test size
            self.assertEqual(len(test),16-train_sizes[folds])
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        # make sure iterator does the correct number of iterations
        self.assertEqual(folds,3)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

    def test_Stratified_testfrac(self):
        # test StratifiedCV train/test split
        # Some test data
        #TODO: handle the case where number of positve classes < folds
        test_data = pandas.DataFrame({
            'key1':['a','a','a','b','b','b','c','c','c','d','d','e','f','g','h','h'],
            'data':range(16),
            'y':[0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1]
        })
        yt = test_data['y']
        # create CVbyKey class with 25% test set
        Z = NewPartition(16,yt=yt,cv_method='StratifiedCV')
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=0)
        self.assertEqual(cv.__class__.__name__,'StratifiedCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # check results
        folds = 0
        test_rows = []
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # train = 75%, test = 25%
            self.assertEqual(len(train),12)
            self.assertEqual(len(test),4)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(sorted(test))
            # check that positive classes are distributed correctly
            self.assertEqual(test_data['y'][test].sum(),2)
            folds += 1
        # make sure cv only generates one train/test set
        self.assertEqual(folds,1)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),4)
        # test all() method
        self.assertEqual(len(set(cv.all().tolist())),16)
        # test len() method
        self.assertEqual(len(cv),1)
        # test new random seed produces different results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=2)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(sorted(test))
        #TODO: fix this
        self.assertNotEqual(test_rows,test_rows2)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=0.25,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(sorted(test))
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=.25,random_state=0,samplesize=12)
        test_rows = []
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(train),9)
            # sample size shouldn't change test size
            self.assertEqual(len(test),4)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(sorted(test))
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

        # test more validation_pct
        for validation_pct in [0.05, 0.15, 0.5, 0.6, 0.8]:
            cv = Z.get_cv(size=len(yt),yt=yt,validation_pct=validation_pct,random_state=0)
            for train,test in cv:
                # check right size
                self.assertTrue(len(train) + len(test) == len(yt))
                # check that rows don't appear in both train & test
                self.assertTrue(set(train) & set(test) == set())

    def test_User_train_test_split(self):
        # test User defined CV train/test split
        test_data = pandas.DataFrame({
            'user_keys':['T','V','T','T','T','T','V','T','T','T','V','V','T','T','V','T'],
            'data':range(16),
            'y':range(16)
        })
        # create CVbyKey class with 25% test set
        Z = NewPartition(16,cv_info=test_data['user_keys'],cv_method='UserCV')
        yt = test_data['y']
        cv = Z.get_cv(size=len(yt),yt=yt,random_state=0)
        self.assertEqual(cv.__class__.__name__,'UserCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # check results
        folds = 0
        test_rows = []
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # train = 75%, test = 25%
            self.assertEqual(len(train),11)
            self.assertEqual(len(test),5)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        # make sure cv only generates one train/test set
        self.assertEqual(folds,1)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),5)
        # test all() method
        self.assertEqual(cv.all().tolist(),range(16))
        # test len() method
        self.assertEqual(len(cv),1)
        # test same random seed produces same results
        cv = Z.get_cv(size=len(yt),yt=yt,random_state=0)
        test_rows2 = []
        for train,test in cv:
            test_rows2.extend(test)
        self.assertEqual(test_rows,test_rows2)

        # test sample size
        cv = Z.get_cv(size=len(yt),yt=yt,random_state=0,samplesize=12)
        test_rows = []
        for train,test in cv:
            # Check partition sizes
            self.assertEqual(len(train),8)
            # sample size shouldn't change test size
            self.assertEqual(len(test),5)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
        # sample size 12 test rows should be the same as sample size 16
        self.assertEqual(test_rows,test_rows2)

        # test User defined CV train/test split with custom levels
        test_data = pandas.DataFrame({
            'user_keys':['TT','VV','TT','TT','TT','TT','VV','TT','TT','TT','VV','VV','TT','TT','VV','TT'],
            'data':range(16),
            'y':range(16)
        })
        # create CVbyKey class with 25% test set
        Z = NewPartition(16,cv_info=test_data['user_keys'],cv_method='UserCV')
        yt = test_data['y']
        cv = Z.get_cv(size=len(yt),yt=yt,random_state=0,training_level='TT',validation_level='VV')
        self.assertEqual(cv.__class__.__name__,'UserCV')
        # make sure class has not been altered
        #self.assertEqual(hmac.new(inspect.getsource(cv.__class__)).hexdigest(),"d314324e8f8d32b1c05dc62b9ed32614")
        # check results
        folds = 0
        test_rows = []
        for train,test in cv:
            # check index sets are ndarrays
            self.assertIsInstance(train,np.ndarray)
            self.assertIsInstance(test,np.ndarray)
            self.assertEqual(train.dtype,np.dtype('int64'))
            self.assertEqual(test.dtype,np.dtype('int64'))
            # train = 75%, test = 25%
            self.assertEqual(len(train),11)
            self.assertEqual(len(test),5)
            # check that rows don't appear in both train & test
            self.assertTrue(set(train) & set(test) == set())
            test_rows.extend(test)
            folds += 1
        # make sure cv only generates one train/test set
        self.assertEqual(folds,1)
        # make sure each row was used only once
        self.assertEqual(len(set(test_rows)),5)
        # test all() method

    def test_missing_partition(self):
        # test User defined CV train/test split with missing values
        test_data = pandas.DataFrame({
            'user_keys':['TT','VV','TT','TT','TT','TT','VV',float('nan'),'TT','TT','VV','VV','TT','TT','VV','TT'],
            'data':range(16),
            'y':range(16)
        })
        # create CVbyKey class with 25% test set
        Z = NewPartition(16,cv_info=test_data['user_keys'],cv_method='UserCV')
        yt = test_data['y']
        cv = Z.get_cv(size=len(yt),yt=yt,random_state=0,training_level='TT',validation_level='VV')
        self.assertEqual(cv.ddr.tolist(),[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1])

    def test_parallel_object_creation(self):
        # make sure RNG seeds work even when run in parallel
        expected = [
         [18,  1, 19,  8, 10, 17,  6, 13,  4,  2,  5, 14,  9,  7, 16, 11,  3, 0, 15, 12],
         [ 3, 16,  6, 10,  2, 14,  4, 17,  7,  1, 13,  0, 19, 18,  9, 15,  8, 12, 11,  5],
         [12,  4, 18,  0,  9,  5,  3, 10,  1, 17,  7, 16, 14,  2, 11,  6, 19, 13, 15,  8],
         [14,  2,  1, 17,  4, 16,  6,  7, 15, 12,  9, 11, 19, 18, 13,  5,  0, 8,  3, 10],
         [19,  3, 18,  6, 13,  4,  0, 17, 12, 11, 15, 10,  9,  2, 16,  7,  8, 1,  5, 14],
         [ 2,  5, 17, 19, 12,  1, 11, 10, 13, 18,  7,  4,  8,  9,  0, 16,  6, 15, 14,  3],
         [17,  7,  5,  6,  8,  2, 15, 14,  4, 11, 12,  1, 18,  0, 16, 13, 19, 3,  9, 10],
         [ 1, 17,  2,  5, 11,  0, 18,  6, 13, 19, 10, 14,  8, 16,  9, 12,  7, 3,  4, 15],
         [ 7, 11,  6, 15, 18,  4, 12,  2, 13,  1, 16,  0, 14, 19,  8, 10,  5, 9, 17,  3],
         [ 3,  6,  7,  5, 16, 15,  2,  4,  9, 10, 18, 11, 14,  0, 12, 13, 19, 17,  8,  1]
        ]
        for i in range(10):
            p = multiprocessing.Pool(processes=10)
            cvs = p.map(get_Z,range(10))
            p.close()
            p.join()
            for i,j in enumerate(cvs):
                self.assertEqual(j.cv.randseq.tolist(),expected[i])

if __name__=='__main__':
    unittest.main()
