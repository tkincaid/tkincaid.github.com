import unittest
import pytest
from mock import Mock, patch

from bson import ObjectId

import common.engine.metrics as metrics
from ModelingMachine.metablueprint.mb8_7 import MBSeriesEight
from ModelingMachine.metablueprint.ref_model_mixins import RReferenceModelsMixin

from tests.ModelingMachine.test_base_mb import BaseTestMB

class TestRRefModelsMixin(BaseTestMB):

    class MB(RReferenceModelsMixin, MBSeriesEight):
        def initial_blueprints(self):
            return []

        def get_metric_for_models(self, metric):
            return metrics.RMSE

    def setUp(self):
        super(TestRRefModelsMixin, self).setUp()
        self.persistent.destroy(table='model_code')

    def test_can_construct_sensibly(self):
        pid = self.make_fake_project()
        mb = self.MB(pid, None)
        self.assertEqual(pid, mb.pid)

    def test_creates_r_reference_models(self):
        print self.MB.__mro__
        rtype = 'Binary'

        pid = self.make_fake_project(task_type=rtype, nunique_y=2)
        one_ref_model = {
            'hash' : '697d4987b39c738b75fb9b19383111ba8523b7b0',
            'name' : 'user model 1',
            'task_id' : ObjectId('52f98f583d587f5713733247'),
            'created' : '2014-07-25 02:47:52.711002',
            'modelpredict' : 'function(model,data) {\n  library(gbm);\n  drop = c();\n  for(i in names(data)) {\n    if(typeof(data[[i]]) == \'character\') {\n      data[[i]] = as.factor(data[[i]]);\n      if(length(levels(data[[i]])) > 1024) {\n        drop = c(drop, i);\n      }\n    }\n  }\n  data = data[,!names(data) %in% drop]\n  predict.gbm(model, data, n.trees=200, type=\'response\');\n}\n',
            'modelfit' : 'function(response,data) {\n  library(gbm);\n  drop = c();\n  for(i in names(data)) {\n    if(typeof(data[[i]]) == \'character\') {\n      data[[i]] = as.factor(data[[i]]);\n      if(length(levels(data[[i]])) > 1024) {\n        drop = c(drop, i);\n      }\n    }\n  }\n  data = data[,!names(data) %in% drop]\n  gbm.fit(data, response, distribution=\'gaussian\', n.trees=200, interaction.depth=10, shrinkage=0.01, bag.fraction=0.5, keep.data=FALSE, verbose=FALSE);\n}\n',
            'type' : 'modeling',
            'rtype': rtype,
            'repository': 'oss',
            'modeltype': 'R',
            'modelsource': '',
            'classname': '',
        }
        self.persistent.create(table='model_code', values=one_ref_model)
        mb = self.MB(pid, None, reference_models=True)
        jobs = mb.next_steps()
        ref_models = [j for j in jobs if j.get('reference_model')]
        self.assertEqual(len(ref_models), 1)


