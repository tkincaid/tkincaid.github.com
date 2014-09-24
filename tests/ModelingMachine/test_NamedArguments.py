###############################################################
#
#   test for NamedArguments class
#
#   Author: Sergey Yurgenson
#
#   Copyright DataRobot Inc. 2013
#
###############################################################

import unittest
from ModelingMachine.engine.tasks.base_modeler import NamedArguments

class dummy_Class(NamedArguments):
    arguments = {
            'A1':{'name':'A1', 'type':'select', 'values':[False,True], 'default':'0'},
	    'A2':{'name':'A2', 'type':'float', 'values':[0.0,1.0], 'default':'0.5'},
            'A3':{'name':'A3', 'type':'select', 'values':['cat','dog'], 'default':'0'},
            'A4':{'name':'A4', 'type':'int', 'values':[-1,1], 'default':'0'},
            'A5':{'name':'A5', 'type':'int', 'values':[0,1000], 'default':'0'},
            'A6':{'name':'A6', 'type':'multi', 'values':{'int':[0,1000],'select':[False, True]},'default':'False'}
            }

    def __init__(self, args=''):
        self.parameters = self.parse_args(args)
        #self.ridit_input=self.parameters['ridit_input']



class test_NamedArguments(unittest.TestCase):
    """ Test suite for NamedArgument class
    """
    def test_parse_args(self):
        """ test the parse_arg function of the class """
        # test value
	task = dummy_Class('A1=1')
	A=task.parameters['A1']
	self.assertEqual(A, True)

        # test value
	task = dummy_Class('A1=False')
	A=task.parameters['A1']
	self.assertEqual(A, False)

        # test default
	task = dummy_Class()
	A=task.parameters['A1']
	self.assertEqual(A, False)

        # test special case parsing
	task = dummy_Class('A1=True')
	A=task.parameters['A1']
	self.assertEqual(A, True)

        # test incorrect value
	self.assertRaises(ValueError,dummy_Class,'A1=5.4')

        # test value
	task = dummy_Class('A2=0.4')
	A=task.parameters['A2']
	self.assertEqual(A, 0.4)
        # test default
	task = dummy_Class()
	A=task.parameters['A2']
	self.assertEqual(A, 0.5)
        # test incorrect value
        #no enforcement of float , SY does not know if it is intentionally
	#self.assertRaises(ValueError,dummy_Class,'A2=5.4')


        # test value
	task = dummy_Class('A3=dog')
	A=task.parameters['A3']
	self.assertEqual(A, 'dog')
        # test value
	task = dummy_Class('A3=1')
	A=task.parameters['A3']
	self.assertEqual(A, 'dog')
        # test default
	task = dummy_Class()
	A=task.parameters['A3']
	self.assertEqual(A, 'cat')
        # test incorrect value
	self.assertRaises(ValueError,dummy_Class,'A3=elefant')


        # test value
	task = dummy_Class('A4=-1')
	A=task.parameters['A4']
	self.assertEqual(A, -1)
        # test default
	task = dummy_Class()
	A=task.parameters['A4']
	self.assertEqual(A, 0)
        # test incorrect value
        #no enforcement of int , SY does not know if it is intentionally
	#self.assertRaises(ValueError,dummy_Class,'A4=10')

        # test multi value
	task = dummy_Class('A6=10')
	A=task.parameters['A6']
	self.assertEqual(A, 10)

        # test multi value
	task = dummy_Class('A6=True')
	A=task.parameters['A6']
	self.assertEqual(A, True)

        # test multi value
	task = dummy_Class('A6=False')
	A=task.parameters['A6']
	self.assertEqual(A, False)

        # test multi value
	task = dummy_Class('A6=0')
	A=task.parameters['A6']
	self.assertEqual(A, 0)


if __name__ == '__main__':
    unittest.main()

