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

from ModelingMachine.metablueprint.preprocessing_list_V5 import *
from ModelingMachine.engine.utilities import validate_blueprint

from common.services.flippers import FLIPPERS

class Test_preprocessing_list(unittest.TestCase):

    def test_valid_bp(self):
        '''
        test if bps are valid
        '''
        model = 'GBC'
        preprocessing_list = [
                DRT_NI_x_OE_COUNT_x_TF,
                DRT_NI_DIFF_RATIO_x_CRED_COUNT_x_TF,
                DRT_NI_DIFF_RATIO_x_OE_CRED_COUNT_x_TF,
                DRN_NI_x_HOT_x_RTM,
                DRN_RDT_x_HOT_x_RTM,
                DRN_NI_x_HOT_x_TF,
                DRN_RDT_x_HOT_x_TF,
                DRN_BT_x_HOT_x_TF,
                DRN_DSC_x_HOT_x_TF,
                DRN_PCA_x_HOT_x_TF,
                DRN_RDT_x_HOT2_x_TF,
                DRN_BT_x_HOT2_x_TF,
                DRG_BT_x_HOT_CRED_x_TF,
                DRG_BT2W_x_HOT_CRED_x_TF,
                DRG_NI_x_OE_CRED_x_TF_x_AS,
                ]
        available_types_list = [('N'), ('C'), ('T'),
                                ('N', 'C'),
                                ('N', 'T'),
                                ('C', 'T'),
                                ('N', 'C', 'T')]
        rtype_list=['Binary', 'Regression', 'TwoStage']
        selection_type_list=[None, 'L1', 'L2', 'RF']
        Y_transf_list = ['', 'logy']
        add_char_list = [False, True]
        add_txtsvm_list = [False, True]
        dtm_args_list = [1, 2]
        validates =[]
        for rtype in rtype_list:
            for selection_type in selection_type_list:
                for Y_transf in Y_transf_list:
                    for available_types in available_types_list:
                        for preprocessing in preprocessing_list:
                            for add_char in add_char_list:
                                for add_txtsvm in add_txtsvm_list:
                                    for dtm_args in dtm_args_list:
                                        if preprocessing in [DRN_NI_x_HOT_x_RTM, DRN_RDT_x_HOT_x_RTM]:
                                            bp = preprocessing(model, available_types, rtype=rtype,
                                                    selection_type=selection_type, Y_transf=Y_transf,
                                                    adjust_pred='gamma_glm')
                                        else:
                                            bp = preprocessing(model, available_types, rtype=rtype,
                                                    dtm_args=dtm_args, add_char=add_char, add_txtsvm=add_txtsvm,
                                                    selection_type=selection_type, Y_transf=Y_transf,
                                                    adjust_pred='gamma_glm')
                                        if FLIPPERS.graybox_enabled:
                                            bp = bp['blueprint']
                                        validates.append(validate_blueprint(bp, available_types))
        self.assertEqual(set(validates), set([True]))

    def test_valid_recsys_bp(self):
        '''
        test if bps are valid
        '''
        model = 'GBC'
        preprocessing_list = [
                DRR_MOD_COUNT_x_DEV_x_MOD_COUNT_x_TF,
                DRR_MOD_COUNT_x_DEV_x_CRED_COUNT_x_TF,
                DRR_MOD_COUNT_x_NUM_x_HOT_x_TF,
                DRR_MOD_COUNT_x_NUM_x_HOT2w_x_TF,
                DRR_MOD_COUNT_x_NUM_x_HOT2w_COUNT_x_TF,
                DRR_MOD_COUNT_x_BT_x_CRED_COUNT_x_TF,
                DRR_MOD_COUNT_x_NUM_x_CRED_COUNT_x_TF_x_AS
                ]
        available_types_list = [('I', 'U'),
                                ('I', 'U', 'N'),
                                ('I', 'U', 'C'),
                                ('I', 'U', 'T'),
                                ('I', 'U', 'N', 'C'),
                                ('I', 'U', 'N', 'T'),
                                ('I', 'U', 'C', 'T'),
                                ('I', 'U', 'N', 'C', 'T')]
        rtype_list=['Binary', 'Regression']
        selection_type_list=[None, 'L1', 'RF']
        Y_transf_list = ['', 'logy']
        cf_model_list = ['SGDSVD', None]
        validates =[]
        for rtype in rtype_list:
            for selection_type in selection_type_list:
                for Y_transf in Y_transf_list:
                    for available_types in available_types_list:
                        for cf_model in cf_model_list:
                            for preprocessing in preprocessing_list:
                                bp = preprocessing(model, available_types, rtype=rtype,
                                        cf_model=cf_model,
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
        a = DRN_RDT_x_HOT_x_TF('LR1 miny;p=1', available_types=('N','C','T'),
                    rtype='Binary', link_args='l=1', Y_transf='miny;', adjust_pred='')
        b = DRN_NI_x_HOT_x_TF('RIDGE sev', available_types=('N','C','T'),
                    rtype='Regression',Y_transf='sev;', adjust_pred='')

        c = freq_sev_prod(a, b, True)
        self.assertEqual(validate_blueprint(c), True)
        c = freq_sev_prod(a, b)
        self.assertEqual(validate_blueprint(c), True)

        expected ={
              '1': (['NUM'], ['NI'], 'T'),
              '2': (['1'], ['RDT2'], 'T'),
              '3': (['CAT'], ['DM2 sc=10;cm=10000'], 'T'),
              '4': (['TXT'], ['SCTXT cn=0'], 'T'),
              '5': (['4'], ['WNGC miny;num=[1, 2];lc=1;sw=[0,1];nrm=2;midf=10;t_m=LogLoss'], 'S'),
              '6': (['5'], ['LINK l=1'], 'T'),
              '7': (['6'], ['ST'], 'T'),
              '8': (['2', '3', '7'], ['LR1 miny;p=1'], 'S'),
              '9': (['NUM'], ['NI'], 'T'),
              '10': (['9'], ['ST'], 'T'),
              '11': (['CAT'], ['DM2 sc=10;cm=10000'], 'T'),
              '12': (['TXT'], ['SCTXT cn=0'], 'T'),
              '13': (['12'], ['WNGR sev;num=[1, 2];lc=1;sw=[0,1];nrm=2;midf=10;t_m=RMSE'], 'S'),
              '14': (['13'], ['ST'], 'T'),
              '15': (['10', '11', '14'], ['RIDGE sev'], 'S'),
              '16': (['8', '15'], ['FSPROD'], 'P')}

        for key in c.keys():
            for i in range(len(c[key])):
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
