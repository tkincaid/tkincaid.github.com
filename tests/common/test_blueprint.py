import unittest
import json
import pytest

import common.entities.blueprint as blueprint_module
import common.entities.blueprint_diagram as bp_diagrams

# Necessary for testing blueprint features
from ModelingMachine.engine.task_map import task_map, get_dynamic_label
TASK_UI_DATA = task_map.get_ui_data()

@pytest.mark.unit
class TestBlueprintNaming(unittest.TestCase):

    def test_blueprint_id_with_double_digits_keys(self):
        bp = {'1': (['NUM'], ['NI'], 'T'),
             '10': (['9'], ['TOP 10000'], 'T'),
             '11': (['10'], ['GLMB'], 'P'),
             '2': (['1'], ['RDT2'], 'T'),
             '3': (['CAT'], ['CCAT'], 'T'),
             '4': (['3'], ['RDT2'], 'T'),
             '5': (['CAT'], ['CRED dm=1'], 'T'),
             '6': (['5'], ['RDT2'], 'T'),
             '7': (['TXT'], ['TM2', 'LR1'], 'S'),
             '8': (['7'], ['RDT2'], 'T'),
             '9': (['2', '4', '6', '8'], ['LR1'], 'T')}
        #If this call does not throw exception, the test is successful
        blueprint_id = blueprint_module.blueprint_id(bp)

    def test_order_agnostic(self):
        bp1 = {'1':( ['NUM'], ['NI','CCZ'], 'T' ),
               '2':( ['CAT'], ['DM','CCZ'], 'T' ),
               '3':( ['1','2'], ['LR1'], 'P' )}
        bp2 = {'2':( ['NUM'], ['NI','CCZ'], 'T' ),
               '1':( ['CAT'], ['DM','CCZ'], 'T' ),
               '3':( ['1','2'], ['LR1'], 'P' )}

        names1 = blueprint_module.blueprint_to_vertex_names(bp1)
        names2 = blueprint_module.blueprint_to_vertex_names(bp2)

        self.assertEqual(names1['3'], names2['3'])
        self.assertEqual(names1['1'], names2['2'])
        self.assertNotEqual(names1['1'], names2['1'])

    def test_id_with_invalid_blueprint(self):
        bp1 = {'1':( ['NUM'], ['NI','CCZ'], 'T' ),
               '2':( ['CAT'], ['DM','CCZ'], 'T' )}
        self.assertRaises(ValueError, blueprint_module.blueprint_id, bp1)

    def test_get_vertex_name_with_invalid_name(self):
        bp1 = {'1':( ['NUM'], ['NI','CCZ'], 'T' ),
               '2':( ['CAT'], ['DM','CCZ'], 'T' ),
               '3':( ['1','2'], ['LR1'], 'P' )}
        self.assertRaises(ValueError, blueprint_module.blueprint_vertex_name, bp1, '4')

    def test_get_vertex_name(self):
        '''Catches if the naming convention changes.  If so, there are probably
        going to be some unintended side-effects - so you will need to be able
        to index based upon both the old way and your new proposed way'''
        bp1 = {'1':( ['NUM'], ['NI','CCZ'], 'T' ),
               '2':( ['CAT'], ['DM','CCZ'], 'T' ),
               '3':( ['1','2'], ['LR1'], 'P' )}

        name = blueprint_module.blueprint_vertex_name(bp1, '3')
        self.assertEqual(name, 'f1ef0a8668c70c730a525ce89e75510c')

    def test_blueprint_id(self):
        bp1 = {'1':( ['NUM'], ['NI','CCZ'], 'T' ),
               '2':( ['CAT'], ['DM','CCZ'], 'T' ),
               '3':( ['1','2'], ['LR1'], 'P' )}

        name = blueprint_module.blueprint_vertex_name(bp1, '3')
        bid = blueprint_module.blueprint_id(bp1)
        self.assertEqual(name, bid)

    def test_can_differentiate_s_and_t(self):
        bp1 = {'1': ( ['NUM'], ['NI','CCZ'], 'S' ),
               '2': ( ['1'], ['LR1'], 'P' ) }
        bp2 = {'1': ( ['NUM'], ['NI','CCZ'], 'T' ),
               '2': ( ['1'], ['LR1'], 'P' ) }
        name1 = blueprint_module.blueprint_id(bp1)
        name2 = blueprint_module.blueprint_id(bp2)
        self.assertNotEqual(name1, name2)

    def test_can_identify_requirements(self):
        bp1 = {'1': ( ['NUM'], ['NI','CCZ'], 'S' ),
               '2': ( ['1'], ['LR1'], 'P' ) }
        bp2 = {'1': ( ['NUM'], ['NI','CCZ'], 'T' ),
               '2': ( ['1'], ['LR1'], 'P' ) }
        names1 = blueprint_module.blueprint_to_vertex_names(bp1)
        names2 = blueprint_module.blueprint_to_vertex_names(bp2)
        self.assertEqual(names1['1'], names2['1'])

@pytest.mark.unit
class TestBlueprintFeatures(unittest.TestCase):

    def test_blueprint_features_dynamic_names_RFC_extra_trees(self):
        '''Funcitionality moved to blueprint.py; easier to test here'''
        blueprint1 = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=1;c=0'], 'P']}
        out = blueprint_module.blueprint_features(task_map, blueprint1)
        self.assertEqual(out, ['Missing Values Imputed', 'ExtraTrees Classifier (Gini)'])

    @pytest.mark.unit
    def test_blueprint_features_dynamic_names_RFC_entropy(self):
        '''Funcitionality moved to blueprint.py; easier to test here'''
        blueprint2 = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1;'], 'P']}
        out = blueprint_module.blueprint_features(task_map, blueprint2)
        self.assertEqual(out, ['Missing Values Imputed', 'RandomForest Classifier (Entropy)'])

    def test_blueprint_features_dynamic_names_works_with_spaces(self):
        '''Funcitionality moved to blueprint.py; easier to test here'''
        blueprint2 = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC tm=Gamma Deviance;e=0;c=1;'], 'P']}
        out = blueprint_module.blueprint_features(task_map, blueprint2)
        self.assertEqual(out, ['Missing Values Imputed', 'RandomForest Classifier (Entropy)'])

    def test_blueprint_features_dynamic_names_RFC_entropy_log_transformed(self):
        '''Funcitionality moved to blueprint.py; easier to test here'''
        blueprint2 = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC logy;e=0;c=1'], 'P']}
        out = blueprint_module.blueprint_features(task_map, blueprint2)
        self.assertEqual(out, ['Missing Values Imputed',
                               'Log Transformed Response',
                               'RandomForest Classifier (Entropy)'])

    def test_text_mining_bps_have_colnames_cn_0(self):
        ''' Test that text mining bps are correctly named'''
        metaDB = {'varTypeString': 'NCTT',
                  'columns': [
                      [0, 'num1', 0],
                      [1, 'cat1', 0],
                      [2, 'first text column', 0],
                      [3, 'second text column', 0]]
                  }
        blueprint = {'1': (['TXT'], ['SCTXT cn=0'], 'T'),
                     '2': (['NUM'], ['NI'], 'T'),
                     '3': (['1'], ['WNGR'], 'P')}
        out = blueprint_module.blueprint_features(task_map, blueprint, metaDB)
        self.assertEqual(out, ['Converter for Text Mining',
                               'Missing Values Imputed',
                               'Auto-Tuned Word N-Gram Text Modeler - first text column'])

    def test_text_mining_bps_have_colnames_cn_1(self):
        ''' Test that text mining bps are correctly named'''
        metaDB = {'varTypeString': 'NTTC',
                  'columns': [
                      [0, 'num1', 0],
                      [1, 'first text column', 0],
                      [2, 'second text column', 0],
                      [3, 'cat1', 0]]
                  }
        blueprint = {'1': (['TXT'], ['SCTXT nonsense;cn=1;'], 'T'),
                     '2': (['1'], ['WNGR'], 'P')}
        out = blueprint_module.blueprint_features(task_map, blueprint, metaDB)
        self.assertEqual(out, ['Converter for Text Mining',
                               'Auto-Tuned Word N-Gram Text Modeler - second text column'])

@pytest.mark.unit
class TestBlueprintBlackboxDiagram(unittest.TestCase):

    def recursive_no_ridit(self, node):
        if 'children' not in node or len(node['children']) == 0:
            return True
        if 'RDT2' not in node['tasks']:
            return all([self.recursive_no_ridit(i) for i in node['children']])
        return False

    def test_simple_diagram_should_rid_converter(self):
        bp = {'1': [['NUM'], ['NI'], 'T'],
              '2': [['1'], ['RFC'], 'P']}
        dg = blueprint_module.blackbox_diagram(bp, TASK_UI_DATA, 'Occult', get_dynamic_label)
        self.assertIn('taskMap', dg)
        self.assertNotIn('NI', dg['taskMap'])
        self.assertEqual('0', dg['id'])
        self.assertEqual(['START'], dg['tasks'])
        self.assertEqual('start', dg['type'])

    def test_graybox_diagram_for_insights_shouldnt_die(self):
        model = 'RIDGE'
        dg = blueprint_module.graybox_diagram_and_feats_for_insight_model(model,
            TASK_UI_DATA, get_dynamic_label)
        self.assertIn('diagram', dg)
        dia = json.loads(dg['diagram'])
        self.assertIn('features', dg)
        self.assertTrue(self.recursive_no_ridit(dia))

    def test_lying_hides_data(self):
        bp = {'1': [['NUM'], ['NI'], 'T'],
              '2': [['1'], ['RFC'], 'P']}
        lies = {'NI': {'label': 'Nothing Interesting'}}
        dg = blueprint_module.get_frontend_diagram_data(
                bp, TASK_UI_DATA, get_dynamic_label, lies)
        self.assertIn('taskMap', dg)
        labels = [dg['taskMap'][key]['label'] for key in dg['taskMap'].keys()]
        self.assertIn('Nothing Interesting', labels)
        self.assertNotIn('Missing Values Imptued', labels)

    def test_ridit_censoring_removes_ridit(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['CAT'], ['CCAT'], 'T'],
                     '3': [['2'], ['RDT2'], 'T'],
                     '4': [['1', '3'], ['RFC'], 'P']}
        bp_out = blueprint_module.ridit_censored(blueprint)

        # Asserts that it is a valid blueprint
        bp_id = blueprint_module.blueprint_id(bp_out)

        diagram = blueprint_module.get_frontend_diagram_data(bp_out,
                TASK_UI_DATA, get_dynamic_label)
        self.assertTrue(self.recursive_no_ridit(diagram))

    def test_ridit_censoring_keeps_both_parents(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['CAT'], ['CCAT'], 'T'],
                     '3': [['2', '1'], ['RDT2'], 'T'],
                     '4': [['3'], ['RFC'], 'P']}

        bp_out = blueprint_module.ridit_censored(blueprint)

        # Asserts that it is a valid blueprint
        bp_id = blueprint_module.blueprint_id(bp_out)

        diagram = blueprint_module.get_frontend_diagram_data(bp_out,
                TASK_UI_DATA, get_dynamic_label)
        self.assertTrue(self.recursive_no_ridit(diagram))

    def test_get_frontend_diagram_data(self):
        '''This test was created for bug #2772 where the front end was showing
        incorrect parameters We found that get_frontend_diagram_data was
        inserting the wrong aliases since it expected keys in the blueprint
        dictionary to be sorted.
        '''

        blueprint = {
            "1" : [["NUM"],["NI"],"T"],
            "2" : [["CAT"],["ORDCAT"],"T"],
            "3" : [["1","2"],["GBC"],"P"]
        }

        diagram = blueprint_module.get_frontend_diagram_data(blueprint,
                TASK_UI_DATA, get_dynamic_label)

        gbc_node = bp_diagrams.find_in_tree_by(diagram, '3', 'id')

        # Assert the task name is replaced with the vertex ID
        self.assertIn('3', gbc_node['tasks'])

    def test_ridit_censoring_keeps_both_children(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RDT2'], 'T'],
                     '3': [['2'], ['ST'], 'T'],
                     '4': [['2'], ['GS'], 'T'],
                     '5': [['3', '4'], ['RFC'], 'P']}

        bp_out = blueprint_module.ridit_censored(blueprint)

        # Asserts that it is a valid blueprint
        bp_id = blueprint_module.blueprint_id(bp_out)

        diagram = blueprint_module.get_frontend_diagram_data(bp_out,
                TASK_UI_DATA, get_dynamic_label)
        self.assertTrue(self.recursive_no_ridit(diagram))

    def test_calib_blueprint_uses_correct_modelname(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC logy'], 'S'],
                     '3': [['2'], ['CALIB'], 'P']}

        dia = bp_diagrams.blackbox_diagram(blueprint,
            TASK_UI_DATA, 'Test Preprocessing', get_dynamic_label)
        self.assertIn('RFC', dia['taskMap'])

    def test_failing_blueprint(self):
        blueprint = {
            u'1': [[u'NUM'], [u'NI'], u'T'],
            u'2': [[u'1'], [u'BTRANSF dist=2;d=2'], u'T'],
            u'3': [[u'2'], [u'ST'], u'T'],
            u'4': [[u'3'], [u'LR1 p=0'], u'T'],
            u'5': [[u'4'], [u'GLMB'], u'P']}
        dia = bp_diagrams.blackbox_diagram(blueprint,
            TASK_UI_DATA, 'Test Preprocessing', get_dynamic_label)
        self.assertEqual(dia['taskMap']['GLMB']['label'],
                         'Generalized Linear Model (Bernoulli Distribution)')

    def test_for_regression(self):
        blueprint = {
            u'1': [[u'NUM'], [u'NI'], u'T'],
            u'2': [[u'1'], [u'BTRANSF dist=2;d=1'], u'T'],
            u'3': [[u'2'], [u'ST'], u'T'],
            u'4': [[u'3'], [u'LR1 p=0'], u'P']}
        dia = bp_diagrams.blackbox_diagram(blueprint,
            TASK_UI_DATA, 'Test Preprocessing', get_dynamic_label)
        self.assertEqual(dia['taskMap']['LR1']['label'],
                         'Regularized Logistic Regression (L1)')

    def test_blackbox_with_calib_keeps_calib(self):
        blueprint = {
            u'1': [[u'NUM'], [u'NI'], u'T'],
            u'2': [[u'1'], [u'BTRANSF dist=2;d=1'], u'T'],
            u'3': [[u'2'], [u'ST'], u'T'],
            u'4': [[u'3'], [u'LR1 logy;p=0'], u'S'],
            u'5': [[u'4'], [u'CALIB'], u'P']}
        dia = bp_diagrams.blackbox_diagram(blueprint,
            TASK_UI_DATA, 'Test Preprocessing', get_dynamic_label)
        labels = [i['label'] for i in dia['taskMap'].values()]
        self.assertIn('Calibrate predictions', labels)


class TestBlueprintBranches(unittest.TestCase):

    @pytest.mark.unit
    def test_bp_branches_single_branch(self):
        """ Test that bp_branches handles a single branch correctly
        """
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1'], 'P']}
        out = blueprint_module.get_bp_branches(blueprint)
        self.assertEqual(out, {'1': {'root': ['NUM'], 'node': 'response',
                                     'tasks': ['1', '2']}})

    @pytest.mark.unit
    def test_bp_branches_multiple_branches(self):
        """Test that bp_branches handles a two>one tree correctly
        """
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['CAT'], ['DM2'], 'T'],
                     '3': [['1', '2'], ['RFC e=0;c=1'], 'P']}
        out = blueprint_module.get_bp_branches(blueprint)
        self.assertEqual(out, {'1': {'root': ['NUM'], 'node': '3',
                                     'tasks': ['1']},
                               '2': {'root': ['CAT'], 'node': '3',
                                     'tasks': ['2']},
                               '3': {'root': ['1', '2'], 'node': 'response',
                                      'tasks': ['3']}})

    @pytest.mark.unit
    def test_bp_branches_many_branches(self):
        blueprint = {'11': (['10'], ['TOP 10000'], 'T'),
                     '10': (['2', '4', '6', '9'], [u'LASSO t_m=Gini'], 'T'),
                     '12': (['11'], ['GLMG '], 'P'),
                     '1': (['NUM'], ['NI'], 'T'),
                     '3': (['CAT'], ['CCAT'], 'T'),
                     '2': (['1'], ['RDT2'], 'T'),
                     '5': (['CAT'], ['CRED dm=1'], 'T'),
                     '4': (['3'], ['RDT2'], 'T'),
                     '7': (['TXT'], ['TM2'], 'T'),
                     '6': (['5'], ['RDT2'], 'T'),
                     '9': (['8'], ['RDT2'], 'T'),
                     '8': (['7'], [u'RIDGE t_m=Gini'], 'S')}
        out = blueprint_module.get_bp_branches(blueprint)
        self.assertEqual(out, {
            '1': {'root': ['NUM'], 'node': '5', 'tasks': ['1', '2']},
            '2': {'root': ['CAT'], 'node': '5', 'tasks': ['3', '4']},
            '3': {'root': ['CAT'], 'node': '5', 'tasks': ['5', '6']},
            '4': {'root': ['TXT'], 'node': '5', 'tasks': ['7', '8', '9']},
            '5': {'root': ['1', '2', '3', '4'],
                  'node': 'response',
                  'tasks': ['10', '11', '12']}})

    @pytest.mark.unit
    def test_bp_branches_fsprod_regression_test(self):
        """ Regression test for Frequency severity modeling
        """
        blueprint = {
            '1': (['CAT'], ['DM2 sc=10;cm=10000'], 'T'),
            '2': (['TXT'], ['TM2 '], 'T'),
            '3': (['2'], ['LR1 miny;p=1'], 'S'),
            '4': (['3'], ['LINK l=1'], 'T'),
            '5': (['4'], ['ST'], 'T'),
            '6': (['1', '5'], ['LR1 miny;p=1'], 'P'),
            '7': (['CAT'], ['DM2 sc=10;cm=10000'], 'T'),
            '8': (['TXT'], ['TM2 '], 'T'),
            '9': (['8'], ['RIDGE sev'], 'S'),
            '10': (['9'], ['ST'], 'T'),
            '11': (['7', '10'], ['RIDGE sev'], 'P'),
            '12': (['6', '11'], ['FSPROD'], 'P')}
        out = blueprint_module.get_bp_branches(blueprint)
        reference = {
            '1': {'root': ['CAT'], 'node': '5', 'tasks': ['1']},
            '2': {'root': ['TXT'], 'node': '5', 'tasks': ['2', '3', '4', '5']},
            '3': {'root': ['CAT'], 'node': '6', 'tasks': ['7']},
            '4': {'root': ['TXT'], 'node': '6', 'tasks': ['8', '9', '10']},
            '5': {'root': ['1', '2'], 'node': '7', 'tasks': ['6']},
            '6': {'root': ['3', '4'], 'node': '7', 'tasks': ['11']},
            '7': {'root': ['5', '6'], 'node': 'response', 'tasks': ['12']}}

        self.assertEqual(reference.keys(), out.keys())
        for key in out.keys():
            self.assertEqual(out[key], reference[key])


class TestBlacklistDetection(unittest.TestCase):
    '''One of the strategies we'll pursue to keep some of our techniques
    in reserve is to collapse whole branches and give them an accurate
    yet deliberately vague name
    '''

    @pytest.mark.unit
    def test_simple_detection_case(self):
        blueprint = {'1': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '2': [['1'], ['BTRANSF d=1'], 'T'],
                     '3': [['2'], ['ST'], 'T'],
                     '4': [['3'], ['GLMG'], 'P']}
        pattern = ['CAT', 'CCAT cmin=33', 'BTRANSF d=1', 'ST']
        where = blueprint_module.find_pattern(blueprint, pattern)

        # Returns a list of dict with these keys
        out = where[0]
        self.assertIn('start', out)
        self.assertEqual(out['start'], '1')

        self.assertIn('finish', out)
        self.assertEqual(out['finish'], '4')

        self.assertIn('tasks', out)
        self.assertEqual(out['tasks'], ['1', '2', '3'])

    @pytest.mark.unit
    def test_find_when_rule_doesnot_begin_from_vertex_1(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '3': [['2'], ['BTRANSF d=1'], 'T'],
                     '4': [['3'], ['ST'], 'T'],
                     '5': [['1', '4'], ['GLMG'], 'P']}
        pattern = ['CAT', 'CCAT cmin=33', 'BTRANSF d=1', 'ST']
        where = blueprint_module.find_pattern(blueprint, pattern)

        out = where[0]
        self.assertEqual(out['start'], '2')
        self.assertEqual(out['finish'], '5')
        self.assertEqual(out['tasks'], ['2', '3', '4'])

    @pytest.mark.unit
    def test_not_found_case(self):
        blueprint = {'1': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '2': [['1'], ['BTRANSF d=1'], 'T'],
                     '3': [['2'], ['ST'], 'T'],
                     '4': [['3'], ['GLMG'], 'P']}
        pattern = ['CAT', 'DM2']
        out = blueprint_module.find_pattern(blueprint, pattern)
        self.assertIsNone(out)

    @pytest.mark.unit
    def test_short_branch_case(self):
        blueprint = {'1': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '2': [['1'], ['BTRANSF d=1'], 'T'],
                     '3': [['2'], ['ST'], 'T'],
                     '4': [['3'], ['GLMG'], 'P']}
        pattern = ['CAT', 'CCAT cmin=33']
        where = blueprint_module.find_pattern(blueprint, pattern)

        out = where[0]
        self.assertEqual(out['start'], '1')
        self.assertEqual(out['finish'], '2')
        self.assertEqual(out['tasks'], ['1'])

    @pytest.mark.unit
    def test_apply_rule_makes_a_substitution(self):
        blueprint = {'1': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '2': [['1'], ['BTRANSF d=1'], 'T'],
                     '3': [['2'], ['ST'], 'T'],
                     '4': [['3'], ['GLMG'], 'P']}
        pattern = ['CAT', 'CCAT cmin=33', 'BTRANSF d=1', 'ST']
        rule ={'pattern': pattern, 'code': 'XCAT1'}
        out = blueprint_module.apply_substitution_rule(blueprint, rule)

        expected= {'1': [['CAT'], ['XCAT1'], 'T'],
                   '2': [['1'], ['GLMG'], 'P']}
        self.assert_blueprint_equal(expected, out)

    @pytest.mark.unit
    def test_apply_rule_when_doesnt_apply_leaves_blueprint_intact(self):
        blueprint = {'1': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '2': [['1'], ['BTRANSF d=1'], 'T'],
                     '3': [['2'], ['ST'], 'T'],
                     '4': [['3'], ['GLMG'], 'P']}
        pattern = ['CAT', 'DM2']
        rule ={'pattern': pattern, 'code': 'XCAT1'}

        out = blueprint_module.apply_substitution_rule(blueprint, rule)
        self.assert_blueprint_equal(blueprint, out)

    @pytest.mark.unit
    def test_apply_rule_catches_multiple_of_same_rule(self):
        blueprint = {'1': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '2': [['1'], ['BTRANSF d=1'], 'T'],
                     '3': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '4': [['3'], ['RDT2'], 'T'],
                     '5': [['2', '4'], ['RFC'], 'P']}
        pattern = ['CAT', 'CCAT cmin=33']
        rule = {'pattern': pattern, 'code': 'XCAT2'}

        out = blueprint_module.apply_substitution_rule(blueprint, rule)
        for k, v in out.items():
            self.assertNotIn('CCAT cmin=33', v[1], 'Substitution missed one')

    def assert_blueprint_equal(self, bp1, bp2):
        '''Some use tuples, some use lists, so this checks if two bps are
        functionally identical
        '''
        self.assertEqual(set(bp1.keys()), set(bp2.keys()))
        for key in bp1.keys():
            v1 = bp1[key]
            v2 = bp2[key]

            self.assertEqual(v1[0], v2[0], 'Differing inputs')
            self.assertEqual(v1[1], v2[1], 'Differing tasks')
            self.assertEqual(v1[2], v2[2], 'Differing operations')

    @pytest.mark.unit
    def test_find_pattern_doesnot_alter_blueprint(self):
        blueprint = {
            '1': (['CAT'], ['ORDCAT '], 'T'),
            '2': [['CAT'], ['CCAT cmin=33'], 'T'],
            '3': (['TXT'], ['TM2 '], 'T'),
            '4': (['3'], ['RIDGE '], 'S'),
            '5': (['NUM'], ['NI'], 'T'),
            '6': [['1', '2', '4', '5'],
                  ['GBR logy;lr=0.05;n=1000;mf=0.5;md=[1, 3, 5];t_m=RMSE'],
                  'P']}
        pattern = ['CAT', 'ORDCAT']
        out = blueprint_module.find_pattern(blueprint, pattern)

        # Was swapping 4 and 5 out for 3 and 3
        self.assertIn('4', blueprint['6'][0])
        self.assertIn('5', blueprint['6'][0])


@pytest.mark.unit
class TestBlueprintViz (unittest.TestCase):

    def setUp(self):
        self.plan = []
        BLUEPRINT = {
                "1": [["NUM"], ["NI"], "T"], #vertex 1, inputs = ['NUM'], tasks = ['NI']
                "2": [['1'], ["RFI"], "P"]   #vertex 2, inputs = ['1'], tasks = ['RFI']
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["NI"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['1','2'], ["RFI"], "P"]
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['1'], ["ST","LR1"], "P"]
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['1','2'], ["ST","LR1"], "P"]
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['1','2'], ["ST","LR1"], "T"],
                "4": [['3'], ["PCA","SVCR"], "P"]
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['TXT'], ["TM","LR1"], "P"],
                "4": [['1','2','3'], ["RFI"], "P"],
                "5": [['1','2','3'], ["GBCT"], "P"],
                "6": [['4','5'], ["GAMG"], "P"]
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['TXT'], ["TM","LR1"], "P"],
                "4": [['1','2','3'], ["ST","LR1"], "T"],
                "5": [['4'], ["PCA","SVCR"], "P"],
                "6": [['1','2','3'], ["RFI"], "P"],
                "7": [['5','6'], ["GAMP"], "P"]
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['TXT'], ["TM","LR1"], "P"],
                "4": [['1','2','3'], ["ST","LR1"], "T"],
                "5": [['4'], ["PCA","SVCR"], "P"],
                "6": [['1','2','3'], ["RFI"], "P"],
                "7": [['1','2','3'], ["GBCT"], "P"],
                "8": [['5','6','7'], ["GAMP"], "P"]
                }
        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['TXT'], ["TM","LR1"], "P"],
                "4": [['1','2','3'], ["ST","LR1"], "T"],
                "5": [['4'], ["PCA","SVCR"], "P"],
                "6": [['1','2','3'], ["ST","LR1"], "P"],
                "7": [['1','2','3'], ["RFI"], "P"],
                "8": [['1','2','3'], ["GBCT"], "P"],
                "9": [['5','6','7','8'], ["GAMG"], "P"]
                }

        self.plan+=[BLUEPRINT]
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['TXT'], ["TM 123 ABC","LR1 ARG1 123"], "P"],
                "4": [['1','2','3'], ["ST ABC","LR1 123456 ABC"], "T"],
                "5": [['4'], ["PCA ABCDEF","SVCR XYZ "], "P"],
                "6": [['1','2','3'], ["ST ABCDEF","LR1"], "P"],
                "7": [['1','2','3'], ["RFI 123 VWXYZ ABC"], "P"],
                "8": [['1','2','3'], ["GBCT VWXYZ"], "P"],
                "9": [['5','6','7','8'], ["GAMB ABC"], "P"]
                }

        self.plan+=[BLUEPRINT]

        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ["DM"], "T"],
                "3": [['1','2'], ["ST","LR1"], "T"],
                "4": [['3'], ["PCA","SVCR"], "P"],
                "5": [['2'], ["NI"], "P"],
                "6": [['4','5'], ["RFI","LR1"], "P"]
                }
        self.plan+=[BLUEPRINT]
        # new and more complex test blueprint added on 4/16/13
        BLUEPRINT = {
                "1": [["NUM"], ["LS"], "T"],
                "2": [['CAT'], ['DM'], 'T'],
                "3": [['TXT'], ['TM','LR1'], 'T'],
                '4': [['1','2','3'],['ST','LR1'], 'T'],
                '5': [['4'],['CCZ','RDT','ST','PCA','SVCR'], 'P'],
                '6': [['4'],['CCZ','RDT','ST','LR1'], 'P'],
                '7': [['NUM'], ['NI'], 'T'],
                '8': [['2','3','7'],['RFI'],'P'],
                '9': [['2','3','7'],['GBCT'], 'P'],
                '10': [['5','6','8','9'],['GAMG'], 'P']
                }

        self.plan+=[BLUEPRINT]



    def test_is_in_tree(self):
        d = dict(id= '1', tasks=['A', 'B'], name='someName', type='someType', children = [{'id':'2', 'name':'child1', 'type':'type1', 'tasks':['C']},
                                                                       {'id':'3','name':'child21', 'type':'type2', 'tasks':['D'],  'children':[{
                                                                                                                                     'id':'4', 'name':'child1', 'type':'type1', 'tasks':['E']
                                                                                                                                     }]
                                                                        }])
        vId = '1'
        self.assertIsNotNone(bp_diagrams.is_in_tree(d, vId));

        vId = '2'
        self.assertIsNotNone(bp_diagrams.is_in_tree(d, vId))

        vId = '3'
        self.assertIsNotNone(bp_diagrams.is_in_tree(d, vId))

        vId = '4'
        self.assertIsNotNone(bp_diagrams.is_in_tree(d, vId))

        vId = '5'
        self.assertIsNone(bp_diagrams.is_in_tree(d, vId))

        tasks = 'C'
        self.assertIsNone(bp_diagrams.is_in_tree(d, tasks))

        tasks = ['C']
        self.assertIsNotNone(bp_diagrams.is_in_tree(d, tasks))

        tasks = ['A', 'B']
        self.assertIsNotNone(bp_diagrams.is_in_tree(d, tasks))

        return

    def test_add_child_to_node(self):
        tree = dict(id= '1')
        vertex =  [["NUM"], ["LS"], "T"]
        id ='2'
        bp_diagrams.add_child_to_node(tree,vertex,id)

        node = bp_diagrams.is_in_tree(tree, id);
        self.assertIsNotNone(node)

        vertex =  [['CAT'], ["DM"], "T"]
        id ='3'
        bp_diagrams.add_child_to_node(node,vertex,id)

        self.assertIsNotNone(node)

    def test_blueprint_to_tree_0(self):
        #===========================================================================
        # BLUEPRINT = {
        #    "1": [["NUM"], ["NI"], "T"], #vertex 1, inputs = ['NUM'], tasks = ['NI']
        #    "2": [['1'], ["RFI"], "P"]   #vertex 2, inputs = ['1'], tasks = ['RFI']
        #    }
        #===========================================================================
        bp = bp_diagrams.blueprint_to_tree(self.plan[0])

        id = "1"
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertIsNotNone(node)

        children = node.get('children', None)
        self.assertIsNotNone(children)

        id = "2"
        node = bp_diagrams.is_in_tree(node, id);
        self.assertIsNotNone(node)

        children = node.get('children', None)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0]['tasks'], ['END'])

        return

    def test_blueprint_to_tree_3(self):
       #============================================================================
       # BLUEPRINT = {
       #     "1": [["NUM"], ["LS"], "T"],
       #     "2": [['CAT'], ["DM"], "T"],
       #     "3": [['1','2'], ["ST","LR1"], "T"],
       #     "4": [['3'], ["PCA","SVCR"], "P"]
       #     }
       #============================================================================
        bp = bp_diagrams.blueprint_to_tree(self.plan[3])

        id = "1"
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertIsNotNone(node)

        children = node.get('children', None)
        self.assertIsNotNone(children)


        id = "3"
        thirdNode = bp_diagrams.is_in_tree(node, id);
        self.assertIsNotNone(node)

        id = "2"
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertIsNotNone(node)

        children = node.get('children', None)
        self.assertIsNotNone(children)

        id = "3"
        thirdNode = bp_diagrams.is_in_tree(node, id);
        self.assertIsNotNone(node)

        id = "4"
        fourthNode = bp_diagrams.is_in_tree(thirdNode, id);
        self.assertIsNotNone(node)

        return

    def test_node_types(self):
        #=======================================================================
        # BLUEPRINT = {
        # "1": [["NUM"], ["LS"], "T"],
        # "2": [['CAT'], ["DM"], "T"],
        # "3": [['TXT'], ["TM","LR1"], "P"],
        # "4": [['1','2','3'], ["ST","LR1"], "T"],
        # "5": [['4'], ["PCA","SVCR"], "P"],
        # "6": [['1','2','3'], ["ST","LR1"], "P"],
        # "7": [['1','2','3'], ["RFI"], "P"],
        # "8": [['1','2','3'], ["GBCT"], "P"],
        # "9": [['5','6','7','8'], ["GAMB"], "P"]
        # }
        #=======================================================================
        bp = bp_diagrams.blueprint_to_tree(self.plan[8])

        id = '0'
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertIsNotNone(node)
        self.assertEqual(node['type'],'start')

        id = '-2'
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertIsNotNone(node)
        self.assertEqual(node['type'],'input')

        id = '5'
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertIsNotNone(node)
        self.assertEqual(node['type'],'task')

        id = '10'
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertIsNotNone(node)
        self.assertEqual(node['type'],'end')

        return

    def test_input_fields(self):
        #=======================================================================
        # BLUEPRINT = {
        # "1": [["NUM"], ["LS"], "T"],
        # "2": [['CAT'], ["DM"], "T"],
        # "3": [['TXT'], ["TM","LR1"], "P"],
        # "4": [['1','2','3'], ["ST","LR1"], "T"],
        # "5": [['4'], ["PCA","SVCR"], "P"],
        # "6": [['1','2','3'], ["ST","LR1"], "P"],
        # "7": [['1','2','3'], ["RFI"], "P"],
        # "8": [['1','2','3'], ["GBCT"], "P"],
        # "9": [['5','6','7','8'], ["GAMG"], "P"]
        # }
        #=======================================================================

        # Check inputs to which a pre-processing step was added
        bp = bp_diagrams.blueprint_to_tree(self.plan[8])
        id = '1'
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertTrue(bp_diagrams.get_prestep_id(id) in node['inputs'])

        id = '2'
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertTrue(bp_diagrams.get_prestep_id(id) in node['inputs'])

        id = '3'
        node = bp_diagrams.is_in_tree(bp, id);
        self.assertTrue(bp_diagrams.get_prestep_id(id) in node['inputs'])

        # Check othe inputs to ensure they were not affected
        id = '4'
        node = bp_diagrams.is_in_tree(bp, id);
        expected_input_ids= ['1','2','3']
        # set <= other, Test whether every element in the set is in other.
        self.assertTrue(set(expected_input_ids) <= set(node['inputs']))
        return

    def test_remove_duplicate_nodes(self):
         #duplicate nodes
        #=======================================================================
        # BLUEPRINT = {
        # "1": [["NUM"], ["LS"], "T"],
        # "2": [['CAT'], ["DM"], "T"],
        # "3": [['TXT'], ["TM","LR1"], "P"],
        # "4": [['1','2','3'], ["ST","LR1"], "T"],
        # "5": [['4'], ["PCA","SVCR"], "P"],
        # "6": [['1','2','3'], ["ST","LR1"], "P"],
        # "7": [['1','2','3'], ["RFI"], "P"],
        # "8": [['1','2','3'], ["GBCT"], "P"],
        # "9": [['5','6','7','8'], ["GAMB"], "P"]
        # }
        #=======================================================================
        bp = self.plan[8]
        self.assertEqual(len(bp.keys()), 9)
        self.assertTrue('6' in bp.keys())
        self.assertTrue('6' in bp['9'][0])

        bp = bp_diagrams.remove_duplicates(bp)

        self.assertEqual(len(bp.keys()), 11)
        self.assertFalse('6' in bp.keys())
        self.assertTrue('4' in bp['9'][0])

    def test_remove_duplicate_inputs(self):
        #duplicate inputs
        #=======================================================================
        # BLUEPRINT = {
        # "1": [["NUM"], ["LS"], "T"],
        # "2": [['CAT'], ['DM'], 'T'],
        # "3": [['TXT'], ['TM','LR1'], 'T'],
        # '4': [['1','2','3'],['ST','LR1'], 'T'],
        # '5': [['4'],['CCZ','RDT','ST','PCA','SVCR'], 'P'],
        # '6': [['4'],['CCZ','RDT','ST','LR1'], 'P'],
        # '7': [['NUM'], ['NI'], 'T'],
        # '8': [['2','3','7'],['RFI'],'P'],
        # '9': [['2','3','7'],['GBCT'], 'P'],
        # '10': [['5','6','8','9'],['GAMB'], 'P']
        # }
        #=======================================================================
        bp = self.plan[11]
        self.assertTrue('NUM' in bp['1'][0])
        self.assertTrue('NUM' in bp['7'][0])

        bp = bp_diagrams.remove_duplicates(bp)

        prestep_id = bp_diagrams.get_prestep_id('1')
        self.assertTrue(prestep_id in bp['1'][0])
        self.assertTrue(prestep_id in bp['7'][0]) #  updated, no more duplicates

        return

    def test_is_int_on_int(self):
        self.assertTrue(blueprint_module.is_int(1))

    def test_is_int_on_int_str(self):
        self.assertTrue(blueprint_module.is_int('1'))

    def test_is_int_on_not_int(self):
        self.assertFalse(blueprint_module.is_int('a'))

    def test_is_prestep_id(self):
        self.assertFalse(bp_diagrams.is_prestep_id('1'))
        self.assertFalse(bp_diagrams.is_prestep_id(1))
        self.assertFalse(bp_diagrams.is_prestep_id(['A','B']))
        self.assertTrue(bp_diagrams.is_prestep_id('-1'))
        self.assertTrue(bp_diagrams.is_prestep_id('-2'))
        return

    def test_remove_duplicates_same_tasks_different_inputs(self):
        BLUEPRINT = {
                '1' : (['NUM'],['LS','CCZ'],'T'),
                '2' : (['CAT'],['DM','CCZ'],'T'),
                '3' : (['TXT'],['TM','LR1'],'T'),
                '4' : (['2'],['ST','LR1'],'T'),
                '5' : (['1','3','4'],['ST','LR1'],'P')
                }
        rb = bp_diagrams.remove_duplicates(BLUEPRINT)
        self.assertIn('5', rb)


    def test_END_with_duplicate_task_lists(self):
        BLUEPRINT = {
                '1' : (['NUM'],['LS','CCZ'],'T'),
                '2' : (['CAT'],['DM','CCZ'],'T'),
                '3' : (['TXT'],['TM','LR1'],'T'),
                '4' : (['2'],['ST','LR1'],'T'),
                '5' : (['1','3','4'],['ST','LR1'],'P')
                }
        x = bp_diagrams.blueprint_to_tree(BLUEPRINT)
        node_5 = bp_diagrams.find_in_tree_by(x, '5', 'id')
        inputset = node_5['inputs']
        self.assertEqual(set(inputset), set(BLUEPRINT['5'][0]))
        for in_node_id in inputset:
            parent_node = bp_diagrams.find_in_tree_by(x, in_node_id, 'id')
            self.assertTrue(bp_diagrams.find_in_tree_by(parent_node, '5', 'id'))


    def test_blueprint_to_tree_on_blender_bp(self):
        bp = {
            '1': [[u'b67aee14ac38a5bd0a8498abe3cb278e'], [u'MEDBL'], u'P']}
        x = bp_diagrams.blueprint_to_tree(bp)
        #TODO assert something, I guess


    def test_blueprint_to_tree_with_custom_input_nodes(self):
        bp = {
            '1': [['DATAPREP'], ['MODEL1'], 'T'],
            '2': [['DATAPREP'], ['MODEL2'], 'T'],
            '3': [['DATAPREP'], ['MODEL3'], 'T'],
            '4': [['1', '2', '3'], ['MEDBL'], 'P']}
        custom_inputs = ['DATAPREP']
        tree = bp_diagrams.blueprint_to_tree(bp, custom_inputs)
        self.assertEqual(tree['id'], '0')  # Just asserts a tree was made

    def test_blueprint_to_tree_with_aliases(self):
        '''Making changes without breaking anything is a delicate dance'''
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['RFC'], 'P']}
        alias = {'RFC': '2'}
        root = bp_diagrams.blueprint_to_tree(bp, aliases=alias)
        target = root['children'][0]['children'][0]['children'][0]
        self.assertEqual(target['tasks'], ['2'])



class TestFindVariableSelections(unittest.TestCase):

    def test_should_find_a_variable_selection(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['RFC'], 'T'],
            '3': [['2'], ['SVMC'], 'P']}
        result = blueprint_module.find_variable_selections(bp, task_map)
        self.assertEqual(len(result), 1)
        self.assertIn('2', result)

    def test_should_return_empty_list_if_no_var_selection(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['SVMC'], 'P']}
        result = blueprint_module.find_variable_selections(bp, task_map)
        self.assertEqual(len(result), 0)

class TestBlueprintPaths(unittest.TestCase):

    def test_simple_chain_returns_one_path(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['PPCA'], 'T'],
            '3': [['2'], ['BTRANSF'], 'T'],
            '4': [['3'], ['MARST'], 'T'],
            '5': [['4'], ['RFC'], 'P']}
        result = blueprint_module.blueprint_paths(bp, '1', '5')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ['1', '2', '3', '4', '5'])

    def test_two_paths_both_returned(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['RFC'], 'T'],
            '3': [['1'], ['GBM'], 'T'],
            '4': [['2', '3'], ['SVMC'], 'P']}
        result = blueprint_module.blueprint_paths(bp, '1', '4')
        self.assertEqual(len(result), 2)


class TestLeastCommonDescendant(unittest.TestCase):

    def test_only_common_is_sink(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['PPCA'], 'T'],
            '3': [['1'], ['BTRANSF'], 'T'],
            '4': [['2'], ['MARST'], 'T'],
            '5': [['3'], ['RFC'], 'T'],
            '6': [['4', '5'], ['SVMC'], 'P']}
        result = blueprint_module.least_common_descendant(bp, '2', '3')

        self.assertEqual(result, '6')

    def test_when_common_is_before_sink(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['PPCA'], 'T'],
            '3': [['1'], ['BTRANSF'], 'T'],
            '4': [['2'], ['MARST'], 'T'],
            '5': [['3', '4'], ['RFC'], 'T'],
            '6': [['5'], ['SVMC'], 'P']}
        result = blueprint_module.least_common_descendant(bp, '2', '3')

        self.assertEqual(result, '5')


class TestBlenderDiagram(unittest.TestCase):

    def test_all_models_are_in_diagram(self):
        input_models = ['3', '5', '7']
        model_codes = ['SVMC', 'RFC', 'LR1']
        blend_code = 'GAMB'
        result = bp_diagrams.blender_diagram(input_models, model_codes,
                                             blend_code, TASK_UI_DATA,
                                             get_dynamic_label)
        labels = [result['taskMap'][key]['label']
                  for key in result['taskMap'].keys()]

        for i in 3, 5, 7:
            self.assertIn('PREP_{}'.format(i), result['taskMap'])
        for code in model_codes:
            label = TASK_UI_DATA[code]['label']
            if label == 'DYNAMIC':
                label = get_dynamic_label(code, None)
            self.assertIn(label, labels)

    def test_existing_glm_blenders_arent_broken(self):
        input_models = ['3', '5', '7']
        model_codes = ['SVMC', 'RFC', 'LR1']
        blend_code = 'RGLM'
        result = bp_diagrams.blender_diagram(input_models, model_codes,
                                             blend_code, TASK_UI_DATA,
                                             get_dynamic_label)
        labels = [result['taskMap'][key]['label']
                  for key in result['taskMap'].keys()]

        for i in 3, 5, 7:
            self.assertIn('PREP_{}'.format(i), result['taskMap'])
        for code in model_codes:
            label = TASK_UI_DATA[code]['label']
            if label == 'DYNAMIC':
                label = get_dynamic_label(code, None)
            self.assertIn(label, labels)


class TestBlueprintVertex(unittest.TestCase):

    def test_init_has_all_params(self):
        vert = blueprint_module.BlueprintVertex('2', ['1'], 'RFC logy', 'P')
        self.assertEqual(vert.idx, '2')
        self.assertEqual(vert.inputs, ['1'])
        self.assertEqual(vert.task, 'RFC logy')
        self.assertEqual(vert.task_code, 'RFC')
        self.assertEqual(vert.task_args_str, 'logy')
        self.assertEqual(vert.task_args, {'logy': '1'})
        self.assertEqual(vert.operation, 'P')

    def test_alt_constructor_has_all_params(self):
        vert = blueprint_module.BlueprintVertex.from_spec(
            '2', [['1'], ['RFC logy'], 'P'])
        self.assertEqual(vert.idx, '2')
        self.assertEqual(vert.inputs, ['1'])
        self.assertEqual(vert.task, 'RFC logy')
        self.assertEqual(vert.task_code, 'RFC')
        self.assertEqual(vert.task_args_str, 'logy')
        self.assertEqual(vert.task_args, {'logy': '1'})
        self.assertEqual(vert.operation, 'P')

    def test_to_spec_behaves_as_expected(self):
        spec = [['1'], ['RFC logy'], 'P']
        vert = blueprint_module.BlueprintVertex.from_spec('2', spec)

        self.assertEqual(vert.to_spec(), spec)

class TestBlueprintModel(unittest.TestCase):

    def test_init_works_out(self):
        bp_spec = {'1': [['NUM'], ['NI'], 'T'],
                   '2': [['1'], ['RFC e=0;c=1;sev'], 'P'],
                   '3': [['1'], ['LR1 miny'], 'P'],
                   '4': [['2', '3'], ['FSPROD'], 'P']}
        blue = blueprint_module.Blueprint(bp_spec)
        self.assertEqual(blue.to_dict(), bp_spec)
        for idx, vtx in blue.vertices.iteritems():
            self.assertIsInstance(vtx, blueprint_module.BlueprintVertex)
            self.assertEqual(vtx.to_spec(), bp_spec[idx])


    def test_last_vertex_with_CALIB(self):
        bp_spec = {'1': [['NUM'], ['NI'], 'T'],
                   '2': [['1'], ['RFC e=0;c=1'], 'S'],
                   '3': [['2'], ['CALIB'], 'P']}
        blue = blueprint_module.Blueprint(bp_spec)
        lastv = blue.last_vertex()
        self.assertEqual(lastv.task_code, 'CALIB')

    def test_modeling_vertex_with_CALIB_isnot_calib(self):
        bp_spec = {'1': [['NUM'], ['NI'], 'T'],
                   '2': [['1'], ['RFC e=0;c=1'], 'S'],
                   '3': [['2'], ['CALIB'], 'P']}
        blue = blueprint_module.Blueprint(bp_spec)
        model_v = blue.model_vertex()
        self.assertEqual(model_v.task_code, 'RFC')

    def test_modeling_vertex_without_CALIB_is_last_vert(self):
        bp_spec = {'1': [['NUM'], ['NI'], 'T'],
                   '2': [['1'], ['RFC e=0;c=1'], 'P']}
        blue = blueprint_module.Blueprint(bp_spec)
        model_v = blue.model_vertex()
        self.assertEqual(model_v.task_code, 'RFC')



class TestGetModelNameToDisplay(unittest.TestCase):

    @pytest.mark.unit
    def test_FSPROD_combines_its_inputs_into_its_name(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1;sev'], 'P'],
                     '3': [['1'], ['LR1 miny'], 'P'],
                     '4': [['2', '3'], ['FSPROD'], 'P']}
        outtask, outname = blueprint_module.get_model_name_to_display(task_map, blueprint)
        self.assertEqual(outtask, 'FSPROD')
        self.assertEqual(outname, 'Two-Stage Model - Regularized Logistic '
                'Regression (L2) and RandomForest Classifier (Entropy)')

    @pytest.mark.unit
    def test_CALIB_is_skipped_over(self):
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1'], 'S'],
                     '3': [['2'], ['CALIB'], 'P']}
        outtask, outname = blueprint_module.get_model_name_to_display(task_map, blueprint)
        self.assertEqual(outtask, 'RFC')
        self.assertEqual(outname, 'RandomForest Classifier (Entropy)')

    @pytest.mark.unit
    def test_normal_case_with_dynamic_name(self):
        """ Test that get_model_name_to_display handles RFC final model correctly
        """
        blueprint = {'1': [['NUM'], ['NI'], 'T'],
                     '2': [['1'], ['RFC e=0;c=1'], 'P']}
        outtask, outname = blueprint_module.get_model_name_to_display(task_map, blueprint)
        self.assertEqual(outtask, 'RFC')
        self.assertEqual(outname, 'RandomForest Classifier (Entropy)')


class TestGetModelType(unittest.TestCase):

    def test_get_model_type_no_calib(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['RFC'], 'P']}
        self.assertEqual(blueprint_module.get_model_type(bp), 'RFC')

    def test_get_model_type_calib(self):
        bp = {
            '1': [['NUM'], ['NI'], 'T'],
            '2': [['1'], ['RFC'], 'S'],
            '3': [['2'], ['CALIB'], 'P']}
        self.assertEqual(blueprint_module.get_model_type(bp), 'RFC')


class TestBuildDiagramTaskmap(unittest.TestCase):

    def test_simple_case_not_ruined(self):
        tasks = ['NUM', 'NI', 'RFC e=0']

        taskmap = bp_diagrams.build_diagram_taskmap(tasks, TASK_UI_DATA,
                                                    get_dynamic_label)
        self.assertIn('NUM', taskmap)
        self.assertIn('NI', taskmap)
        self.assertIn('RFC', taskmap)

    def test_simple_case_with_lies(self):
        tasks = ['NUM', 'NI', 'RFC e=0']
        lies = {'NI': {'label': 'Quantum Entanglement Device'}}

        taskmap = bp_diagrams.build_diagram_taskmap(tasks, TASK_UI_DATA,
                                                    get_dynamic_label,
                                                    lies=lies)

        self.assertIn('NI', taskmap)
        self.assertEqual(taskmap['NI']['label'],
                         'Quantum Entanglement Device')

    def test_case_with_code_multiplicity(self):
        tasks = ['NUM', 'NI', 'RFC e=0', 'RFC e=1', 'GLMG']
        aliases = {'RFC e=0': '1',
                   'RFC e=2': '2'}

        taskmap = bp_diagrams.build_diagram_taskmap(tasks, TASK_UI_DATA,
                                                    get_dynamic_label,
                                                    aliases=aliases)

        labels = [v['label'] for v in taskmap.values()]
        self.assertIn('ExtraTrees Classifier (Gini)', labels)
        self.assertIn('RandomForest Classifier (Gini)', labels)


if __name__ == '__main__':
    unittest.main()
