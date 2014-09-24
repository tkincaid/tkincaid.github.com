'''Tests to run on the development (test) metablueprints'''

import unittest
from abc import ABCMeta, abstractmethod
from mock import patch, Mock

from common.entities.blueprint import blueprint_id

import ModelingMachine.engine.metrics as metrics
import ModelingMachine.metablueprint.dev_metablueprints as tm

from common.services.flippers import FLIPPERS


class BaseCaseDevMB(object):

    __metaclass__ = ABCMeta

    @classmethod
    @abstractmethod
    def get_mb_class(cls):
        '''This method is called in each test case to provide the class
        of metablueprint that is under test, for example ``Dscblueprint``
        '''
        pass

    @classmethod
    def setUpClass(cls):
        MetablueprintClass = cls.get_mb_class()
        cls.pid = '5223deadbeefdeadbeeffeed'
        cls.uid = '5223feeddeadbeefdeadbeef'
        cls.dataset_id = '5332deadfadebeefaddd5223'
        cls.tempstore = Mock()
        cls.persistent = Mock()
        with patch.object(MetablueprintClass, '_get_data') as fake_get_data:
            cls.mb = MetablueprintClass(cls.pid, cls.uid, cls.dataset_id,
                                        tempstore=cls.tempstore,
                                        persistent=cls.persistent)

    def setUp(self):
        self.configure_project()

    def configure_project(self, size=1000, metric='RMSE', pct_min_y=0.01,
                          pct_max_y=0.01, min_y=0, max_y=100, nunique_y=500,
                          orig_size=1000, varTypeString='NNNNNN',
                          target_type='Binary'):
        '''Used to set the project and metadata values of the metablueprint.
        If you have a specific scenario to test, you can call this manually
        at the beginning of your test.  This should all be in-memory i.e.
        very fast, so don't worry that it gets called as part of setUp and
        then called manually
        '''
        orig_cols = len(varTypeString)
        self.mb._project = dict(
            target = {'size': size, 'type': target_type},
            metric = metric,
            default_dataset_id = self.dataset_id
            )
        self.mb._metadata = {}
        self.mb._metadata[self.dataset_id] = dict(
            shape=(orig_size, orig_cols),
            varTypeString=varTypeString,
            pct_min_y=pct_min_y,
            pct_max_y=pct_max_y,
            min_y=min_y,
            max_y=max_y,
            nunique_y=nunique_y
            )

    def validate_blueprints(self, blueprints):
        for blueprint in blueprints:
            if FLIPPERS.graybox_enabled:
                if 'blueprint' in blueprint.keys():  # NamedPreprocessor used
                    blueprint = blueprint['blueprint']
            bid = blueprint_id(blueprint)

    def test_generic_case_builds_viable_models(self):
        vartypes = {'N', 'C', 'T'}
        with patch.object(self.mb, 'get_available_types') as fake_vt:
            fake_vt.return_value = vartypes
            blueprints = self.mb.initial_blueprints()

            self.validate_blueprints(blueprints)

    def test_with_many_low(self):
        self.configure_project(pct_min_y=0.4)
        vartypes = {'N', 'C', 'T'}
        with patch.object(self.mb, 'get_available_types') as fake_vt:
            fake_vt.return_value = vartypes
            blueprints = self.mb.initial_blueprints()

            self.validate_blueprints(blueprints)

    def test_with_many_high(self):
        self.configure_project(pct_max_y=0.4)
        vartypes = {'N', 'C', 'T'}
        with patch.object(self.mb, 'get_available_types') as fake_vt:
            fake_vt.return_value = vartypes
            blueprints = self.mb.initial_blueprints()

            self.validate_blueprints(blueprints)

    def test_with_many_low_and_RMSLE(self):
        self.configure_project(pct_min_y=0.4, metric=metrics.RMSLE)
        vartypes = {'N', 'C', 'T'}
        with patch.object(self.mb, 'get_available_types') as fake_vt:
            fake_vt.return_value = vartypes
            blueprints = self.mb.initial_blueprints()

            self.validate_blueprints(blueprints)

    def test_with_many_high_and_RMSLE(self):
        self.configure_project(pct_max_y=0.4, metric=metrics.RMSLE)
        vartypes = {'N', 'C', 'T'}
        with patch.object(self.mb, 'get_available_types') as fake_vt:
            fake_vt.return_value = vartypes
            blueprints = self.mb.initial_blueprints()

            self.validate_blueprints(blueprints)

    def test_really_big(self):
        self.configure_project(size=250000, orig_size=250000)
        vartypes = {'N', 'C', 'T'}
        with patch.object(self.mb, 'get_available_types') as fake_vt:
            fake_vt.return_value = vartypes
            blueprints = self.mb.initial_blueprints()

            self.validate_blueprints(blueprints)

    def test_min_inflated(self):
        self.configure_project(pct_min_y=0.5, target_type='Regression')
        vartypes = {'N', 'C', 'T'}
        with patch.object(self.mb, 'get_available_types') as fake_vt:
            fake_vt.return_value = vartypes
            blueprints = self.mb.initial_blueprints()

            self.validate_blueprints(blueprints)

class TestDscBlueprint(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.Dscblueprint


class TestDscBlueprint2(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.Dscblueprint2


class TestETblueprint(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.ETblueprint


class TestETblueprint2(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.ETblueprint2



class TestL2blueprint(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.L2blueprint


class TestLassoblueprint(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.Lassoblueprint


class TestLassoblueprint2(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.Lassoblueprint2


class TestRFselbluerpint(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.RFselblueprint


class TestSvmKnnblueprint(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.SvmKnnblueprint


class TestSvmKnnblueprint2(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.SvmKnnblueprint2


class TestSvmKnnblueprint3(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.SvmKnnblueprint3

class TestNeuralBlueprints(BaseCaseDevMB, unittest.TestCase):

    @classmethod
    def get_mb_class(cls):
        return tm.NeuralBlueprints
