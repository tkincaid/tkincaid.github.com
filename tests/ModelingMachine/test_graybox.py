import unittest

import ModelingMachine.metablueprint.graybox as gb
from ModelingMachine.engine.task_map import task_map


class TestBlacklistSafeDiagram(unittest.TestCase):

    def test_diagram_func_with_blacklist(self):
        blueprint = {'1': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '2': [['1'], ['BTRANSF d=1'], 'T'],
                     '3': [['2'], ['ST'], 'T'],
                     '4': [['3'], ['GLMG'], 'P']}
        value = gb.blacklist_safe_diagram(blueprint)
        self.assertNotIn('CCAT', value['taskMap'])

    def test_blacklist_for_weirder_blueprint(self):
        blueprint = {'1': (['CAT'], ['ORDCAT '], 'T'),
                     '2': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '3': (['TXT'], ['TM2 '], 'T'),
                     '4': (['3'], ['RIDGE '], 'S'),
                     '5': (['NUM'], ['NI'], 'T'),
                     '6': [['1', '2', '4', '5'],
                           ['GBR logy;lr=0.05;n=1000;mf=0.5;md=[1, 3, 5];t_m=RMSE'],
                           'P']}
        safe = gb.blacklist_safe_blueprint(blueprint)
        final_vertex = safe['5']
        for i in ['1', '2', '3', '4']:
            self.assertIn(i, final_vertex[0])

    def test_blacklist_safe_diagram_for_weirder_blueprint(self):
        blueprint = {'1': (['CAT'], ['ORDCAT '], 'T'),
                     '2': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '3': (['TXT'], ['TM2 '], 'T'),
                     '4': (['3'], ['RIDGE '], 'S'),
                     '5': (['NUM'], ['NI'], 'T'),
                     '6': [['1', '2', '4', '5'],
                           ['GBR logy;lr=0.05;n=1000;mf=0.5;md=[1, 3, 5];t_m=RMSE'],
                           'P']}
        safe = gb.blacklist_safe_diagram(blueprint)
        #Win if this is not erroring

    def test_two_converters_for_cat_get_consolidated(self):

        blueprint = {'1': (['CAT'], ['ORDCAT '], 'T'),
                     '2': [['CAT'], ['CCAT cmin=33'], 'T'],
                     '3': (['TXT'], ['TM2 '], 'T'),
                     '4': (['3'], ['RIDGE '], 'S'),
                     '5': (['NUM'], ['NI'], 'T'),
                     '6': [['1', '2', '4', '5'],
                           ['GBR logy;lr=0.05;n=1000;mf=0.5;md=[1, 3, 5];t_m=RMSE'],
                           'P']}
        dia = gb.blacklist_safe_diagram(blueprint)

        x = diagram_find_task(dia, 'CAT')
        y = diagram_find_node(dia, x)
        self.assertNotIn('ORDCAT', dia['taskMap'])
        self.assertNotIn('CCAT', dia['taskMap'])
        self.assertEqual(len(y), 1)
        self.assertEqual(len(y[0]['children']), 1, 'CAT branches were not '
                'consolidated')

    def test_two_cat_converters_with_extra_step_gets_consolidated(self):
        blueprint = {
            '1': [['CAT'], ['ORDCAT'], 'T'],
            '2': [['CAT'], ['CCAT'], 'T'],
            '3': [['2'], ['RDT2'], 'T'],
            '4': [['1', '3'], ['RFC'], 'P']}

        dia = gb.blacklist_safe_diagram(blueprint)
        x = diagram_find_task(dia, 'CAT')
        y = diagram_find_node(dia, x)
        self.assertEqual(len(y), 1)
        self.assertEqual(len(y[0]['children']), 1, 'CAT branches were not '
                'consolidated')

    def test_this_blueprint_does_not_error(self):
        '''Because it was doing weird stuff before'''
        bp = {'1': (['CAT'], ['DM2 sc=10; cm=10000'], 'T'),
              '2': (['TXT'], ['TM2 '], 'T'),
              '3': (['2'], ['RIDGE logy;'], 'S'),
              '4': (['3'], ['LINK l=3'], 'T'),
              '5': (['4'], ['ST'], 'T'),
              '6': (['NUM'], ['NI'], 'T'),
              '7': (['6'], ['BTRANSF logy;d=1'], 'T'),
              '8': (['7'], ['ST'], 'T'),
              '9': (['1', '5', '8'], ['RIDGE logy;t_m=RMSE'], 'P')}

        dia = gb.blacklist_safe_diagram(bp)
        labels = [dia['taskMap'][key]['label'] for key in dia['taskMap'].keys()]
        self.assertNotIn(task_map.get_label('LINK'), labels)

def diagram_find_task(diagram, task):
    '''Return the id of the node that uses this task'''
    if task in diagram['tasks']:
        return diagram['id']
    children = diagram.get('children', [])
    for child in children:
        found = diagram_find_task(child, task)
        if found is not None:
            return found
    return None

def diagram_find_node(diagram, node_id):
    '''Return all the nodes that have the given id'''
    nodes = []
    if diagram['id'] == node_id:
        nodes.append(diagram)
        return nodes  # Safe to return here because no node can be its
                      # own parent
    children = diagram.get('children', [])
    for child in children:
        new_nodes = diagram_find_node(child, node_id)
        if len(new_nodes) > 0:
            nodes.extend(new_nodes)
            return nodes
    return nodes
