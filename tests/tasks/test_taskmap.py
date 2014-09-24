'''The strings used to designate classes in task_map are tested here.

They need to be loadable to the ModelingMachine in order to do any
modeling.  There are (actually, will be) other tests that test the
lesser feature of the task_map, but this is what verifies that they
are all valid

'''


#########################################################
#
#       Unit Test for task_map
#
#       Author: Tom de Godoy
#
#       Copyright DataRobot, Inc. 2013
#
########################################################

import unittest
import pytest

from ModelingMachine.engine.task_map import task_map, BASE_NODES, split_args, get_dynamic_label
from ModelingMachine.engine.task_map import dynamic_label_RFC, dynamic_label_RFR, dynamic_label_LR, dynamic_label_SVMC, dynamic_label_SVMR
from ModelingMachine.engine.task_map import dynamic_label_KNNC, dynamic_label_KNNR, dynamic_label_GBR, dynamic_label_ESGBR, dynamic_label_DTC
from ModelingMachine.engine.task_map import dynamic_label_GLMR, dynamic_label_ELNR, dynamic_label_GBMCR, dynamic_label_CRF
from ModelingMachine.engine.tasks.converters import BaseConverter
from ModelingMachine.engine.tasks.transformers import BaseTransformer
from ModelingMachine.engine.tasks.base_model_class import BaseModelTransformer
from ModelingMachine.engine.tasks.base_modeler import BaseModeler
from common import load_class
from common.entities.blueprint import BASE_INPUT_TYPES

from ModelingMachine.engine.tasks.rf import RFC, RFR
from ModelingMachine.engine.tasks.logistic import LogRegL1
from ModelingMachine.engine.tasks.svc import SVMC, SVMR
from ModelingMachine.engine.tasks.knn import KNNC, KNNR
from ModelingMachine.engine.tasks.gbm import GBR, ESGBR
from ModelingMachine.engine.tasks.cart import CARTClassifier
from ModelingMachine.engine.tasks.elnet import ElnetR
from ModelingMachine.engine.tasks.glm import GLMR
from ModelingMachine.engine.tasks.cgbm import GBMCR
from ModelingMachine.engine.tasks.crf import CRF


class TestTaskMap(unittest.TestCase):
    def is_valid_class(self,x):
        if issubclass(x,BaseConverter) or issubclass(x,BaseTransformer) or issubclass(x,BaseModelTransformer) or issubclass(x,BaseModeler):
            return True
        else:
            return False

    def is_model(self,item):
        x = item.get('class')
        Task = load_class(x)
        return issubclass(Task,BaseModelTransformer) or issubclass(Task,BaseModeler)

    def test_task_map_contents(self):
        """ checks that each key-value pair in the task_map contains reasonable data
        """
        keys = task_map.keys()
        for key in keys:
            item = task_map.get(key)
            self.assertIsInstance(item, dict,'%s: task value is not a dict'%key)
            self.assertIn('class',item.keys(),'%s: task dict is missing the class'%key)

            Task = load_class(item.get('class'))
            self.assertIn('label',item.keys(),'%s: task dict is missing the label'%key)
            self.assertIn('icon',item.keys(),'%s: task dict is missing the icon'%key)
            self.assertTrue( self.is_valid_class(Task) ,'%s: task class is not a valid class'%key)
            self.assertIsInstance(item.get('label'), str, '%s: task label is not a string'%key)
            inputs = item.get('converter_inputs')
            if issubclass(Task,BaseConverter):
                self.assertIsInstance(inputs,list,'%s: converter input list is not a list'%key)
                self.assertGreaterEqual(len(inputs),1,'%s: converter input list is empty'%key)
                self.assertLessEqual(set(inputs),set(BASE_INPUT_TYPES),'%s: converter input list is invalid'%key)
            else:
                self.assertEqual(inputs,None,'%s: non-converter input list is not None'%key)
            self.assertIn(item.get('icon'),[0,1,2],'%s: icon value not in [0,1,2]'%key)
            if self.is_model(item):
                self.assertIn(item.get('target_type'),['r','p','b'],'%s: model response type not in ["r","p","b"]'%key)
                self.assertNotEqual(item.get('model_family'),None,'%s: model family is missing'%key)
                self.assertIn(
                    item.get('model_family'),
                    ['GLM', 'GLMNET', 'DGLM', 'DGLMNET', 'GAM', 'GBM', 'RF',
                     'SVM', 'KNN', 'BLENDER', 'TS', 'DT', 'NB', 'DUMMY',
                     'OTHER', 'RI', 'NN', 'CAL', 'TEXT', 'FM'],
                    '%s: model family %s not valid' % (key, item.get('model_family')))
            else:
                self.assertIsNone(
                    item.get('target_type'),
                    '%s: non-model response type is not None' % key)
                self.assertIsNone(
                    item.get('model_family'),
                    '%s: non-model model_family value is not None' % key)

    def test_basic_methods(self):
        value = task_map.data['DM2']
        self.assertEqual(task_map['DM2'], value)
        self.assertEqual(task_map.get('DM2'), value)
        self.assertEqual(task_map.get_class('DM2'), load_class(value.get('class')))
        self.assertEqual(task_map.get_label('DM2'), value.get('label'))
        self.assertEqual(task_map.get_input_types('DM2'), value.get('converter_inputs'))
        self.assertEqual(task_map.get_icon('DM2'), value.get('icon'))
        self.assertEqual(task_map.get_target_type('DM2'), value.get('target_type'))
        self.assertEqual(task_map.get_model_family('DM2'), value.get('model_family'))

    def validate_args(self,code):
        args=task_map.get_arguments(code)
        if args==None or len(args)==0:
            return
        self.assertIsInstance(args,dict,'%s: argument is not a dict'%code)
        args_keys = args.keys()
        if type(args_keys[0])==int:
            #positional arguments only, or
            self.assertEqual(set(args_keys),set(range(1,len(args_keys)+1)),'%s: incomplete set of positional arguments'%code)
        else:
            #named arguments only
            self.assertTrue(all(type(i)==str for i in args_keys),'%s: named and positional arguments cannot be mixed'%code)
        for key in args_keys:
            arg = args[key]
            self.assertEqual(set(arg.keys()),set(['name','type','values','default']),'%s, %s: Invalid argument keys'%(code,key))
            self.assertIsInstance(arg['name'],str,'%s: Argument "name" not a string'%code)
            self.assertIn(arg['type'],['int','float','num','minmax','select','intgrid','floatgrid','multi','list'],
                    '%s, %s: Invalid value of argument "type"'%(code,key))
            print 'Arg values: {}'.format(arg['values'])
            print 'It is a {}'.format(type(arg['values']))
            if arg['type'] in ['num','multi']:
                self.assertIsInstance(arg['values'],dict,'%s, %s: Argument "values" should be a dict for this "type"'%(code,key))
            else:
                vals = arg['values']
                self.assertTrue(isinstance(vals, list) or isinstance(vals, tuple),
                                '%s, %s: Argument "values" not a list'%(code,key))

    def test_task_arguments(self):
        self.assertEqual(task_map.get_arguments('DM'),None)
        self.assertNotEqual(task_map.get_arguments('RFI'),None)
        for task in task_map:
            self.validate_args(task)

    def test_task_description(self):
        for task in task_map:
            desc = task_map.get_description(task)
            self.assertIsInstance(desc,dict)
            self.assertEqual(set(desc.keys()), set(['label','description','url','converter_inputs',
                'icon','target_type','model_family','sparse_input','arguments','version']))
            self.assertEqual(task_map.get_arguments(task),desc['arguments'])

    def test_task_parameters(self):
        out = task_map.get_parameters('RFC')
        self.assertIsInstance(out,dict)

        out = task_map.get_parameters('RFC nt=123;ls=12')
        self.assertEqual(out['n_estimators'],'123')
        self.assertEqual(out['min_samples_leaf'],'12')

    def test_base_nodes(self):
        self.assertLess({'START','END','NUM','CAT','TXT'},set(BASE_NODES.keys()))
        for i in BASE_NODES:
            self.assertIsInstance(BASE_NODES[i],dict)
            self.assertEqual(set(BASE_NODES[i].keys()),{'label'})

    def test_get_ui_data(self):
        out = task_map.get_ui_data()
        self.assertIsInstance(out,dict)
        self.assertIn('START',out)
        for i in task_map:
            self.assertEqual(out[i],task_map.get_description(i))
        for i in BASE_NODES:
            for k in BASE_NODES[i]:
                self.assertEqual(out[i][k],BASE_NODES[i][k])

    @pytest.mark.unit
    def test_get_dynamic_label_RFC(self):
        out = get_dynamic_label('RFC', 'e=1;c=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'ExtraTrees Classifier (Gini)')
        out = get_dynamic_label('RFC', 'e=0;c=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'RandomForest Classifier (Entropy)')

    @pytest.mark.unit
    def test_dynamic_label_RFC_ExtraTrees(self):
        out = dynamic_label_RFC('e=1;c=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'ExtraTrees Classifier (Gini)')

    @pytest.mark.unit
    def test_dynamic_label_RFC_RandomForest(self):
        out = dynamic_label_RFC('e=0;c=1')
        self.assertEqual(out, 'RandomForest Classifier (Entropy)')
        out = dynamic_label_RFC('e=0;nt=5000;c=entropy;mf=auto')
        self.assertEqual(out, 'RandomForest Classifier (Entropy)')

    @pytest.mark.unit
    def test_dynamic_label_RFC_Default(self):
        out = dynamic_label_RFC('e=%s'%RFC.arguments['e']['default']+';c=%s'%RFC.arguments['c']['default'])
        self.assertEqual(out, dynamic_label_RFC())

    @pytest.mark.unit
    def test_get_dynamic_label_RFR(self):
        out = get_dynamic_label('RFR', 'e=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'ExtraTrees Regressor')
        out = get_dynamic_label('RFR', 'e=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'RandomForest Regressor')

    @pytest.mark.unit
    def test_dynamic_label_RFR_Default(self):
        out = dynamic_label_RFR('e=%s'%RFR.arguments['e']['default'])
        self.assertEqual(out, dynamic_label_RFR())

    @pytest.mark.unit
    def test_get_dynamic_label_LR(self):
        out = get_dynamic_label('LR1', 'p=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Regularized Logistic Regression (L1)')
        out = get_dynamic_label('LR1', 'p=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Regularized Logistic Regression (L2)')

    @pytest.mark.unit
    def test_dynamic_label_LR_Default(self):
        out = dynamic_label_LR('p=%s'%LogRegL1.arguments['p']['default'])
        self.assertEqual(out, dynamic_label_LR())

    @pytest.mark.unit
    def test_get_dynamic_label_SVMC(self):
        out = get_dynamic_label('SVMC', 'k=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Classifier (Linear Kernel)')
        out = get_dynamic_label('SVMC', 'k=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Classifier (Polynomial Kernel)')
        out = get_dynamic_label('SVMC', 'k=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Classifier (Radial Kernel)')
        out = get_dynamic_label('SVMC', 'k=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Classifier (Sigmoid Kernel)')

    @pytest.mark.unit
    def test_dynamic_label_SVMC_Default(self):
        out = dynamic_label_SVMC('k=%s'%SVMC.arguments['k']['default'])
        self.assertEqual(out, dynamic_label_SVMC())

    @pytest.mark.unit
    def test_get_dynamic_label_SVMR(self):
        out = get_dynamic_label('SVMR', 'k=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Regressor (Linear Kernel)')
        out = get_dynamic_label('SVMR', 'k=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Regressor (Polynomial Kernel)')
        out = get_dynamic_label('SVMR', 'k=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Regressor (Radial Kernel)')
        out = get_dynamic_label('SVMR', 'k=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Support Vector Regressor (Sigmoid Kernel)')

    @pytest.mark.unit
    def test_dynamic_label_SVMR_Default(self):
        out = dynamic_label_SVMR('k=%s'%SVMR.arguments['k']['default'])
        self.assertEqual(out, dynamic_label_SVMR())

    @pytest.mark.unit
    def test_get_dynamic_label_KNNC(self):
        out = get_dynamic_label('KNNC', 'm=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Classifier (Euclidean Distance)')
        out = get_dynamic_label('KNNC', 'm=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Classifier (Manhattan Distance)')
        out = get_dynamic_label('KNNC', 'm=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Classifier (Chebyshev Distance)')
        out = get_dynamic_label('KNNC', 'm=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Classifier (Minkowski Distance)')
        out = get_dynamic_label('KNNC', 'm=4')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Classifier (Weighted Minkowski Distance)')
        out = get_dynamic_label('KNNC', 'k=5;m=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, '5-Nearest Neighbors Classifier (Minkowski Distance)')

    @pytest.mark.unit
    def test_dynamic_label_KNNC_Default(self):
        out = dynamic_label_KNNC('m=%s'%KNNC.arguments['m']['default'])
        self.assertEqual(out, dynamic_label_KNNC())

    @pytest.mark.unit
    def test_get_dynamic_label_KNNR(self):
        out = get_dynamic_label('KNNR', 'm=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Regressor (Euclidean Distance)')
        out = get_dynamic_label('KNNR', 'm=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Regressor (Manhattan Distance)')
        out = get_dynamic_label('KNNR', 'm=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Regressor (Chebyshev Distance)')
        out = get_dynamic_label('KNNR', 'm=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Regressor (Minkowski Distance)')
        out = get_dynamic_label('KNNR', 'k=[15,20];m=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Tuned K-Nearest Neighbors Regressor (Minkowski Distance)')
        out = get_dynamic_label('KNNR', 'm=4')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Regressor (Weighted Minkowski Distance)')
        out = get_dynamic_label('KNNR', 'k=auto;m=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Auto-tuned K-Nearest Neighbors Regressor (Minkowski Distance)')

    @pytest.mark.unit
    def test_dynamic_label_KNNR_Default(self):
        out = dynamic_label_KNNR('m=%s'%KNNR.arguments['m']['default'])
        self.assertEqual(out, dynamic_label_KNNR())

    @pytest.mark.unit
    def test_get_dynamic_label_GBR(self):
        out = get_dynamic_label('GBR', 'l=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor (Least-Squares Loss)')
        out = get_dynamic_label('GBR', 'l=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor (Least-Absolute-Deviation Loss)')
        out = get_dynamic_label('GBR', 'l=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor (Huber Loss)')
        out = get_dynamic_label('GBR', 'l=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor (Quantile Loss)')
        out = get_dynamic_label('GBR', 'l=4')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor (Poisson Loss)')

    @pytest.mark.unit
    def test_dynamic_label_GBR_Default(self):
        out = dynamic_label_GBR('l=%s'%GBR.arguments['l']['default'])
        self.assertEqual(out, dynamic_label_GBR())

    @pytest.mark.unit
    def test_get_dynamic_label_ESGBR(self):
        out = get_dynamic_label('ESGBR', 'l=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor with Early Stopping (Least-Squares Loss)')
        out = get_dynamic_label('ESGBR', 'l=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor with Early Stopping (Least-Absolute-Deviation Loss)')
        out = get_dynamic_label('ESGBR', 'l=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor with Early Stopping (Huber Loss)')
        out = get_dynamic_label('ESGBR', 'l=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor with Early Stopping (Quantile Loss)')
        out = get_dynamic_label('ESGBR', 'l=4')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Gradient Boosted Trees Regressor with Early Stopping (Poisson Loss)')

    @pytest.mark.unit
    def test_dynamic_label_ESGBR_Default(self):
        out = dynamic_label_ESGBR('l=%s'%ESGBR.arguments['l']['default'])
        self.assertEqual(out, dynamic_label_ESGBR())

    @pytest.mark.unit
    def test_get_dynamic_label_DTC(self):
        out = get_dynamic_label('DTC', 'c=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Decision Tree Classifier (Gini)')
        out = get_dynamic_label('DTC', 'c=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Decision Tree Classifier (Entropy)')

    @pytest.mark.unit
    def test_dynamic_label_DTC_Default(self):
        out = dynamic_label_DTC('c=%s'%CARTClassifier.arguments['c']['default'])
        self.assertEqual(out, dynamic_label_DTC())

    @pytest.mark.unit
    def test_get_dynamic_label_GLMR(self):
        out = get_dynamic_label('GLMR', 'd=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Ordinary Least Squares Regression')
        out = get_dynamic_label('GLMR', 'd=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Generalized Linear Model (Poisson Distribution)')
        out = get_dynamic_label('GLMR', 'd=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Generalized Linear Model (Gamma Distribution)')
        out = get_dynamic_label('GLMR', 'd=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Generalized Linear Model (Tweedie Distribution)')

    @pytest.mark.unit
    def test_dynamic_label_GLMR_Default(self):
        out = dynamic_label_GLMR('d=%s'%GLMR.arguments['d']['default'])
        self.assertEqual(out, dynamic_label_GLMR())

    @pytest.mark.unit
    def test_get_dynamic_label_ELNR(self):
        out = get_dynamic_label('ELNR', 'd=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Elastic-Net Regressor (Gaussian Distribution)')
        out = get_dynamic_label('ELNR', 'd=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Elastic-Net Regressor (Poisson Distribution)')
        out = get_dynamic_label('ELNR', 'd=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Elastic-Net Regressor (Gamma Distribution)')
        out = get_dynamic_label('ELNR', 'd=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'Elastic-Net Regressor (Tweedie Distribution)')

    @pytest.mark.unit
    def test_dynamic_label_ELNR_Default(self):
        out = dynamic_label_ELNR('d=%s'%ElnetR.arguments['d']['default'])
        self.assertEqual(out, dynamic_label_ELNR())

    @pytest.mark.unit
    def test_get_dynamic_label_GBMCR(self):
        out = get_dynamic_label('GBMCR', 'd=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'DataRobot Gradient Boosted Trees Regressor (Gaussian Distribution)')
        out = get_dynamic_label('GBMCR', 'd=2')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'DataRobot Gradient Boosted Trees Regressor (Poisson Distribution)')
        out = get_dynamic_label('GBMCR', 'd=3')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'DataRobot Gradient Boosted Trees Regressor (AdaBoost)')

    @pytest.mark.unit
    def test_dynamic_label_GBMCR_Default(self):
        out = dynamic_label_GBMCR('d=%s'%GBMCR.arguments['d']['default'])
        self.assertEqual(out, dynamic_label_GBMCR())

    @pytest.mark.unit
    def test_get_dynamic_label_CRF(self):
        out = get_dynamic_label('CRF', 'rt=0')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'DataRobot RandomForest Classifier')
        out = get_dynamic_label('CRF', 'rt=1')
        self.assertIsInstance(out, basestring)
        self.assertEqual(out, 'DataRobot RandomForest Regressor')

    @pytest.mark.unit
    def test_dynamic_label_CRF_Default(self):
        out = dynamic_label_CRF('rt=%s'%CRF.arguments['rt']['default'])
        self.assertEqual(out, dynamic_label_CRF())

    @pytest.mark.unit
    def test_split_args(self):
        out = split_args('arg1=foo;arg2=bar;foo=foo')
        self.assertIsInstance(out, dict)
        self.assertEqual(out['arg1'], 'foo')
        self.assertEqual(out['arg2'], 'bar')
        self.assertEqual(out['foo'], 'foo')
        out = split_args(None)
        self.assertTrue(out is None)


def test_mandatory_attributes():
    """Checks if task_map entries have mandatory attributes. """
    mandatory_attrs = {'class', 'version', 'label', 'icon'}
    for taskname, attrs in task_map.data.iteritems():
        assert mandatory_attrs.issubset(set(attrs.keys())), \
               'Task %s does not contain the mandatory keys: %r' % (taskname, set(attrs.keys()))

def test_default_constructor():
    """Test if tasks are default constructable. """
    n_errors = 0
    errors = []
    for taskname in task_map.data.iterkeys():
        if taskname == 'NOISE':
            # FXIME NOISE looks broken!
            continue
        try:
            Task = task_map.get_class(taskname)
            Task()
        except:
            n_errors += 1
            errors.append(taskname)

    assert n_errors == 0, 'Default constructor failed for %d tasks: %r' % (n_errors, errors)


if __name__ == '__main__':
    unittest.main()
