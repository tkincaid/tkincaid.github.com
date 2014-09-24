import unittest
from mock import patch

from common.engine.validation import ModelValidation
from ModelingMachine.engine.task_map import task_map
from config.engine import EngConfig

class TestClass(unittest.TestCase):

    def setUp(self):
        self.bp1 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['RGBC'],'P']}
        self.bp2 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['RRFC ls=10;n=100'],'P']}
        self.bp3 = {'1':[['1234'], ['GAMB'], 'P']}
        self.bp4 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GBC'],'P']}
        self.bp5 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['RFC ls=10;n=100'],'P']}
        self.bp6 = {'1':[['1234'], ['GLMB'], 'P']}

        self.bp7 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GBC'],'S'], '3':[['2'],['CALIB e=1'],'P']}
        self.bp8 = {'1':[['NUM'],['NI'],'T'], '2':[['1'],['GBC'],'S'], '3':[['2'],['CALIB e=2'],'P']}

    @patch.dict(EngConfig, {'HANDLE_DISABLED_TASKS': True})
    def test_is_disabled_model_True(self):
        validation = ModelValidation()

        blender = {}
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp1})
        self.assertTrue(out)

        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp2})
        self.assertTrue(out)

        ins =  [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp1,'blender':{}}]
        ins += [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp2,'blender':{}}]
        blender['inputs'] = ins
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp2})
        self.assertTrue(out)

        ins =  [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp4,'blender':{}}]
        ins += [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp5,'blender':{}}]
        blender['inputs'] = ins
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp3})
        self.assertTrue(out)

        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp7})
        self.assertTrue(out)

    @patch.dict(EngConfig, {'HANDLE_DISABLED_TASKS': False})
    def test_handle_disabled_task_flag(self):
        validation = ModelValidation()

        blender = {}
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp1})
        self.assertFalse(out)

        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp2})
        self.assertFalse(out)

        ins =  [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp1,'blender':{}}]
        ins += [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp2,'blender':{}}]
        blender['inputs'] = ins
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp2})
        self.assertFalse(out)

        ins =  [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp4,'blender':{}}]
        ins += [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp5,'blender':{}}]
        blender['inputs'] = ins
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp3})
        self.assertFalse(out)

        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp7})
        self.assertFalse(out)


    @patch.dict(EngConfig, {'HANDLE_DISABLED_TASKS': True})
    def test_is_disabled_model_False(self):
        validation = ModelValidation()

        blender = {}
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp4})
        self.assertFalse(out)

        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp5})
        self.assertFalse(out)

        ins =  [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp4,'blender':{}}]
        ins += [{'dataset_id':'1234','samplepct':100,'blueprint':self.bp5,'blender':{}}]
        blender['inputs'] = ins
        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp6})
        self.assertFalse(out)

        out = validation.is_disabled_model({'blender':blender, 'blueprint':self.bp8})
        self.assertFalse(out)

    @unittest.skip("Skipping because tasks not removed yet")
    @patch.dict(EngConfig, {'HANDLE_DISABLED_TASKS': True})
    def test_taskmap_must_not_contain_disabled_tasks(self):
        validation = ModelValidation()
        #verify that task map does not contain a disabled tasks
        #disabled task names cannot be reused
        for task in task_map:
            self.assertNotIn(task, validation.disabled_tasks)

    @patch.dict(EngConfig, {'HANDLE_DISABLED_TASKS': True})
    def test_flag_disabled_models(self):
        validation = ModelValidation()

        x = [{'blueprint':self.bp1}, {'blueprint':self.bp2}, {'blueprint':self.bp4}, {'blueprint':self.bp5}]
        expected = (True, True, False, False)

        out = validation.flag_disabled_models(x)
        for i,j in zip(out, expected):
            self.assertEqual(set(i.keys()),{'blueprint','disabled'})
            self.assertEqual(i['disabled'], j)

    @patch.dict(EngConfig, {'HANDLE_DISABLED_TASKS': True})
    def test_adv_blender(self):
        validation = ModelValidation()

        request = {'blueprint': {u'1': [[u'3e6af364c612417f1cd3cc4ea70d40d2'], [u'GLMB logitx;sw_b=2'], u'P']}, 
                'samplepct': 70, 'features': [], 'icons': [0], 'blueprint_id': u'816c8a85663032919e2dc637402625a5', 
                'pid': '53a1cb0b3e0fd15146925bec', 'bp': u'1+2+3+4+9+14+15+24', 'model_type': u'Advanced GLM Blender', 
                'dataset_id': u'53a1cb0e3e0fd151dbca826d', 'blender': {u'inputs': [
                    {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'GLMB'], u'S']}, u'dataset_id': u'53a1cb0e3e0fd151dbca826d', u'samplepct': 70, u'blender': {}}, 
                    {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'3': [[u'2'], [u'LR1 p=0'], u'S'], u'2': [[u'1'], [u'ST'], u'T']}, 
                        u'dataset_id': u'53a1cb0e3e0fd151dbca826d', u'samplepct': 70, u'blender': {}}, 
                    {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'3': [[u'2'], [u'LR1 p=1'], u'S'], u'2': [[u'1'], [u'ST'], u'T']}, 
                        u'dataset_id': u'53a1cb0e3e0fd151dbca826d', u'samplepct': 70, u'blender': {}}, 
                    {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'RFC e=0'], u'S']}, 
                        u'dataset_id': u'53a1cb0e3e0fd151dbca826d', u'samplepct': 70, u'blender': {}}, 
                    {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'2': [[u'1'], [u'RRFC2'], u'S']}, 
                        u'dataset_id': u'53a1cb0e3e0fd151dbca826d', u'samplepct': 70, u'blender': {}}, 
                    {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], 
                        u'2': [[u'1'], [u'RFC e=0;t_a=2;t_n=1;t_f=0.15;ls=[5, 10, 20];mf=[0.2, 0.3, 0.4, 0.5];t_m=LogLoss'], u'S']}, 
                        u'dataset_id': u'53a1cb0e3e0fd151dbca826d', u'samplepct': 70, u'blender': {}}, 
                    {u'blueprint': {u'1': [[u'NUM'], [u'NI'], u'T'], u'3': [[u'1'], [u'RATIO dist=2;ivr=0.01'], u'T'], 
                        u'2': [[u'1'], [u'DIFF ivr=0.01'], u'T'], u'5': [[u'4'], [u'RFC e=0;mf=0.3'], u'T t=0.001'], 
                        u'4': [[u'1', u'2', u'3'], [u'BIND'], u'T'], 
                        u'6': [[u'5'], [u'RFC e=1;t_a=2;t_n=1;t_f=0.15;ls=[5, 10, 20];mf=[0.2, 0.3, 0.4, 0.5];t_m=LogLoss'], u'S']}, 
                        u'dataset_id': u'53a1cb0e3e0fd151dbca826d', u'samplepct': 70, 
                        u'blender': {}}]}}

        out = validation.is_disabled_model(request)
        self.assertTrue(out)



if __name__ == '__main__':
    unittest.main()
