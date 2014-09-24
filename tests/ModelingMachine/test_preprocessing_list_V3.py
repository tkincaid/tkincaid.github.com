#########################################################
#
#       test for predefined preprocessing steps
#
#       Author: Xavier Conort
#
#       Copyright DataRobot, Inc. 2014
#
########################################################

import unittest

from ModelingMachine.metablueprint.preprocessing_list_V3 import *
from ModelingMachine.engine.utilities import validate_blueprint

from common.services.flippers import FLIPPERS

class Test_preprocessing_list(unittest.TestCase):

    def test_valid_bp(self):
        '''
        test if bps are valid
        '''
        model = 'GBC'
        preprocessing_list = [
                DRT_NI_x_OE_x_TF,
                DRT_NI_x_OE_COUNT_x_TF,
                DRT_RDT_x_OE_RDTCOUNT_x_RDTTF,
                DRT_RDT_x_RDTOE_RDTCOUNT_x_RDTTF,
                DRT_RDT_x_OE_RDTCOUNT_x_TF,
                DRT_NI_x_CRED_COUNT_x_TF,
                DRN_NI_x_HOT_x_RTM,
                DRN_RDT_x_HOT_x_RTM,
                DRN_BT_x_HOT_x_RTM,
                DRN_NI_x_HOT_x_TF,
                DRN_RDT_x_HOT_x_TF,
                DRN_BT_x_HOT_x_TF,
                DRN_DSC_x_HOT_x_TF,
                DRN_PCA_x_HOT_x_TF,
                DRN_RDT_x_HOT2_x_TF,
                DRN_BT_x_HOT2_x_TF,
                DRN_RDT_x_HOT_x_TF_PCA,
                DRN_RDT_x_HOT_x_RTM_PCA,
                DRG_NI_x_HOT_x_TF,
                DRG_NI_x_CRED_COUNT_x_TF,
                DRG_RDT_x_CRED_COUNT_x_TF,
                DRG_BT_x_CRED_COUNT_x_TF,
                DRG_NI_x_HOT_CRED_x_TF,
                DRG_RDT_x_HOT_CRED_x_TF,
                DRG_BT_x_HOT_CRED_x_TF,
                DRG_AS_x_HOT_CRED_x_TF,
                DRG_NI_x_HOT_COUNT_x_TF,
                DRG_RDT_x_HOT_COUNT_x_TF,
                DRG_BT_x_HOT_COUNT_x_TF
                ]
        available_types_list = [('N'), ('C'), ('T'),
                                ('N', 'C'),
                                ('N', 'T'),
                                ('C', 'T'),
                                ('N', 'C', 'T')]
        rtype_list=['Binary', 'Regression', 'TwoStage']
        selection_type_list=[None, 'L1', 'RF']
        Y_transf_list = ['', 'logy']
        validates =[]
        for rtype in rtype_list:
            for selection_type in selection_type_list:
                for Y_transf in Y_transf_list:
                    for available_types in available_types_list:
                        for preprocessing in preprocessing_list:
                            if preprocessing in [DRN_RDT_x_HOT_x_RTM_PCA, DRN_RDT_x_HOT_x_TF_PCA]:
                                bp = preprocessing(model, available_types, rtype=rtype,
                                        Y_transf=Y_transf)
                            else:
                                bp = preprocessing(model, available_types, rtype=rtype,
                                        selection_type=selection_type, Y_transf=Y_transf,
                                        adjust_pred='gamma_glm')
                            if FLIPPERS.graybox_enabled:
                                bp = bp['blueprint']
                            validates.append(validate_blueprint(bp, available_types))
        self.assertEqual(set(validates), set([True]))


    def test_freq_sev_prod(self):
        '''
        test frequency severity bps are valid
        '''
        a = DRN_BT_x_HOT_x_TF(
                model='LR1 miny;p=1', available_types=('N','C','T'), rtype='Binary',
                btransf_args='dist=2;d=2',
                num_standardize =True, cat_standardize=False, txt_standardize=True,
                selection_type='L1', link_args='l=1', Y_transf='miny;')

        b = DRG_NI_x_HOT_CRED_x_TF(
                model='GLMR sev', available_types=('N','C','T'), rtype='Regression',
                onehot_args='sc=10;cm=32;dc=1', cred_args='cmin=33',
                selection_type=None, link_args='', Y_transf='sev;')


        if FLIPPERS.graybox_enabled:
            a = a['blueprint']
            b = b['blueprint']

        c = freq_sev_prod(a, b)
        self.assertEqual(validate_blueprint(c), True)
        c = freq_sev_prod(a, b, True)
        self.assertEqual(validate_blueprint(c), True)

        expected ={
                '1': (['NUM'], ['NI'], 'T'),
                '2': (['1'], ['BTRANSF miny;dist=2;d=2'], 'T'),
                '3': (['2'], ['ST'], 'T'),
                '4': (['CAT'], ['DM2 sc=10;cm=10000'], 'T'),
                '5': (['TXT'], ['TM2 '], 'T'),
                '6': (['5'], ['LR1 miny;p=1'], 'S'),
                '7': (['6'], ['LINK l=1'], 'T'),
                '8': (['7'], ['ST'], 'T'),
                '9': (['3', '4', '8'], ['LR1 miny;p=0'], 'T'),
                '10': (['9'], ['LR1 miny;p=1'], 'S'),
                '11': (['NUM'], ['NI'], 'T'),
                '12': (['CAT'], ['DM2 sc=10;cm=32;dc=1'], 'T'),
                '13': (['CAT'], ['CRED1 sev;cmin=33'], 'T'),
                '14': (['TXT'], ['TM2 '], 'T'),
                '15': (['14'], ['RIDGE sev;'], 'S'),
                '16': (['11', '12', '13', '15', '10'], ['GLMR sev'], 'P'),
                '17': (['10', '16'], ['FSPROD'], 'P')
        }
        for key in c.keys():
            print key
            for i in range(len(c[key])):
                print i
                self.assertEqual(c[key][i], expected[key][i])

    def test_preprocessing_list_cnbc(self):
        """Test preprocessing for NB. """
        for at in [('N','C','T'), ('N',), ('C',), ('T', ), ('T', 'C',), ('C', 'N')]:
            b = NB_x_HOT_x_TF(available_types=at, rtype='Binary', mi_type='median', ni_args='',
                              onehot_args='sc=10;cm=10000', cat_bayes_args='',
                              dtm_args='', txt_bayes_args='', adjust_pred='')

            self.assertTrue(validate_blueprint(b))

        self.assertRaises(ValueError, NB_x_HOT_x_TF, ('N','C','T'), 'Regression')


class Test_NN_preprocess(unittest.TestCase):

    def test_nn_with_cat_and_text_uses_both(self):
        rtype = 'Binary'
        Ytransf = ''
        available_types = set(['N', 'C'])

        result = DR_NN_DEFAULT_BLUEPRINT(rtype, Ytransf, available_types)
        if FLIPPERS.graybox_enabled:
            blueprint = result['blueprint']
        else:
            blueprint = result
        inputs = [v[0][0] for k, v in blueprint.items()]

        self.assertIn('NUM', inputs)
        self.assertIn('CAT', inputs)

    def test_nn_with_text_ignores_text_if_num_present(self):
        '''Just because text hasn't been tested as much'''
        rtype = 'Binary'
        Ytransf = ''
        available_types = set(['N', 'C', 'T'])

        result = DR_NN_DEFAULT_BLUEPRINT(rtype, Ytransf, available_types)
        if FLIPPERS.graybox_enabled:
            blueprint = result['blueprint']
        else:
            blueprint = result
        inputs = [v[0][0] for k, v in blueprint.items()]

        self.assertIn('NUM', inputs)
        self.assertIn('CAT', inputs)
        self.assertNotIn('TXT', inputs)


    def test_nn_with_just_cat_is_doesnot_use_num(self):
        rtype = 'Regression'
        Ytransf = ''
        available_types = set(['C'])

        result = DR_NN_DEFAULT_BLUEPRINT(rtype, Ytransf, available_types)
        if FLIPPERS.graybox_enabled:
            blueprint = result['blueprint']
        else:
            blueprint = result
        inputs = [v[0][0] for k, v in blueprint.items()]

        self.assertIn('CAT', inputs)
        self.assertNotIn('NUM', inputs)


    def test_nn_with_just_txt_does_make_a_blueprint(self):
        rtype = 'Regression'
        Ytransf = ''
        available_types = set(['T'])

        result = DR_NN_DEFAULT_BLUEPRINT(rtype, Ytransf, available_types)
        if FLIPPERS.graybox_enabled:
            blueprint = result['blueprint']
        else:
            blueprint = result
        inputs = [v[0][0] for k, v in blueprint.items()]

        self.assertIn('TXT', inputs)


if __name__=='__main__':
    unittest.main()
