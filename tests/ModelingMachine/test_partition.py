################################################################
#
#       unit test for Partition
#
#       Author: Tom DeGodoy
#
#       Copyright DataRobot, Inc. 2013
#
################################################################

import unittest
import sys
import os
import numpy as np
import pandas

from itertools import chain

mm = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(mm,'../..'))

from ModelingMachine.engine.partition import Partition
from ModelingMachine.engine.new_partition import NewPartition
from ModelingMachine.engine.tasks.base_modeler import BaseModeler

def plot_partition(part,n=None):
    n = part.size
    out=''
    for p in part:
        for i in range(n):
            if i in part.V(**p):
                out += '+'
            elif i in part.S(**p):
                out += '|'
            elif i in part.T(**p):
                out += '.'
            else:
                out += ' '
        out += '\n'
    return out

def legacy_T(Z, r, k):
    '''
    return training set row indices (of the sample)
    '''
    if r==-1 and k==-1:
        return [i for i in range(Z.size) if Z.randseq[i]<Z.samplesize ]
    elif r==-1:
        tmp = Z.gcv!=k
        return [i for i in range(Z.size) if tmp[i]*(Z.randseq[i]<Z.samplesize)]
    elif k==-1:
        tmp = Z.ddr!=r
        return [i for i in range(Z.size) if tmp[i]*(Z.randseq[i]<Z.samplesize)]
    tmp = (Z.gcv!=k)*(Z.ddr!=r)
    return [i for i in range(Z.size) if tmp[i]*(Z.randseq[i]<Z.samplesize)]

def legacy_V(Z,r,k):
    if k==-1: return []
    if r==-1:
        tmp = Z.gcv==k
        return [i for i in range(Z.size) if tmp[i]*(Z.randseq[i]<Z.samplesize)]
    tmp = (Z.gcv==k)*(Z.ddr!=r)
    return [i for i in range(Z.size) if tmp[i]*(Z.randseq[i]<Z.samplesize)]

def legacy_S(Z,r,k=None,ignore_samplesize=False):
    if r==-1: return []
    tmp = Z.ddr==r
    if ignore_samplesize:
        return [i for i in range(Z.size) if tmp[i]]
    else:
        return [i for i in range(Z.size) if tmp[i]*(Z.randseq[i]<Z.samplesize)]

def legacy_A(Z,r=None,k=None):
    return [i for i in range(Z.size) if Z.randseq[i]<Z.samplesize]



class TestPartitionFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_p100f5r4(self):
        part = Partition(100, folds=5, reps=4, total_size=120)
        self.assertEqual(len(part),20)
        for i in part:
            self.assertEqual(len(part.T(**i)),60)
            self.assertEqual(len(part.V(**i)),15)
            self.assertEqual(len(part.S(**i)),25)

    def test_p100f5r1(self):
        part = Partition(100, folds=5, reps=1, total_size=120)
        self.assertEqual(len(part),5)
        for i in part:
            self.assertEqual(len(part.T(**i)),64)
            self.assertEqual(len(part.V(**i)),16)
            self.assertEqual(len(part.S(**i)),20)
        answer = """|.|+|...|+|.|..+......|.+....|+++.||....|...|+.+.+..|.||.+.|+...|+.....|....+..||....+...+..........
|+|.|.+.|.|.|.........|+..+..|....||.+..|+.+|.....+.|.||+..|.++.|......|.......||.++.......++..+....
|.|.|...|.|+|.+.+....+|.....+|....||....|.+.|.+.....|.||...|...+|.+....|.+....+||+.....+..+.....+.+.
|.|.|+.+|.|.|+....+++.|..+...|...+||+.+.|...|...+...|.||..+|....|......|..+....||.....+.+....+......
|.|.|...|.|.|....+....|....+.|....||...+|...|......+|+||...|....|..++++|+..+.+.||...+.........+..+.+
"""
        self.assertEqual(plot_partition(part,100),answer)

    def test_p100f1r1(self):
        part = Partition(100, folds=1, reps=1,total_size=120)
        self.assertEqual(len(part),1)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(0,80,20))
        answer = "+++||++++|+++++|++++++++|+++++|||++++++++++++|+|+|++|++|+|++|++++|++++++++++|+++|++++|+++|++++++++++\n"
        self.assertEqual(plot_partition(part,100),answer)
        part.set(samplepct=40)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(0,38,10))
        answer = "+ +   +      ++|+++++ + | +        ++   + +++|+|+|  |++ +|+ |+    + + ++    |   |+ ++  +   ++ +   ++\n"
        self.assertEqual(plot_partition(part,100),answer)

    def test_p100f1r1t4(self):
        part = Partition(100, folds=1, reps=1, testfrac=4,total_size=120)
        self.assertEqual(len(part),1)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(0,75,25))
        answer = "++++++|++++|+++++|+|+++++|+++|||+|+++|++++++|++|++++|+|++++|++++++||++++++++|+|++++++++||++|+++|+|+|\n"
        self.assertEqual(plot_partition(part,100),answer)
        part.set(samplepct=30)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(0,27,9))
        answer = "+ +   |      ++++|+|  +   +        ++   +   |++  +  |+| +++ ++    | + ++    |   +      |    + +     \n"
        self.assertEqual(plot_partition(part,100),answer)

    def test_p100f1r1t3(self):
        part = Partition(100, folds=1, reps=1, testfrac=3,total_size=120)
        self.assertEqual(len(part),1)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(0,66,34))
        answer = "++++||++|+++|++|+||+++|+||||++|+|+|||+|+++|++|++++++|+++++++++||+++|+++++++|++|+++|+|++|++|||+||++++\n"
        self.assertEqual(plot_partition(part,100),answer)
        part.set(samplepct=50)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(0, 40, 20))
        answer = "+ ++  + | +  ++|+||++ | |||       |||+  + |++|++++ +|+++++++++ |  + + +++   +   ++ +| +|   || |   ++\n"
        self.assertEqual(plot_partition(part,100),answer)

    def test_p100f5r0(self):
        part = Partition(100, folds=5, reps=0,total_size=100)
        self.assertEqual(len(part),5)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(80,20,0))
        answer = """...++....+.....+........+.....+++............+.+.+..+..+.+..+....+..........+...+....+...+..........
++....+................+..+..+....+..+..++.+......+.....+....++...................++.......++..+....
..+........+..+.+....+......+......+......+...+............+...+..+......+....++.+.....+..+.....+.+.
.....+.+..+.++....+++.+..+.......+..+.+.........+.....+...+...............+...........+.+....+......
........+........+.........+...........+....+......+.+..........+..++++++..+.+......+.........+..+.+
"""
        self.assertEqual(plot_partition(part,100),answer)

    def test_p100f5r5(self):
        part = Partition(size=100,folds=5,reps=5,total_size=100)
        part.set(max_folds=0)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(80,0,20))
        self.assertEqual(len(part),25)
        part.set(max_folds=0,samplepct=80)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(80,0,20))
        self.assertEqual(len(part),25)
        part.set(max_folds=0,samplepct=60)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(60,0,15))
        answer = """| |.  ..| |  .........|....   .  .||....|...|..... .|.||...|....  . . .|.  .. . |. ..... . .. . ....
. .|  ... .  .||..||......|   .  ....|...........| |.|.......|.|  | . ...  .. . .. ...|. . .. | ....
. ..  ... .  |.......|.....   |  .....|..|......|. .....||..|...  . | ...  .. . .| .|..| . |. . .|..
. ..  |.. .  .......|......   .  |..|.....||.|||.. ...........|.  . . |..  |. . .. ..|.. . .. . |..|
. ..  .|. .  ...||.....|||.   .  ......|.......... .......|.....  . . ..|  .| | .. |.... | .| . ..|.
"""
        self.assertEqual(plot_partition(part,100),answer)
        self.assertEqual(len(part),25)
        part.set(max_folds=5,samplepct=100)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(64,16,20))
        part.set(max_reps=1)
        self.assertEqual(len(part),25)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(64,16,20))
        answer = """|.|+|...|+|.|..+......|.+....|+++.||....|...|+.+.+..|.||.+.|+...|+.....|....+..||....+...+..........
|+|.|.+.|.|.|.........|+..+..|....||.+..|+.+|.....+.|.||+..|.++.|......|.......||.++.......++..+....
|.|.|...|.|+|.+.+....+|.....+|....||....|.+.|.+.....|.||...|...+|.+....|.+....+||+.....+..+.....+.+.
|.|.|+.+|.|.|+....+++.|..+...|...+||+.+.|...|...+...|.||..+|....|......|..+....||.....+.+....+......
|.|.|...|.|.|....+....|....+.|....||...+|...|......+|+||...|....|..++++|+..+.+.||...+.........+..+.+
"""
        self.assertEqual(plot_partition(part,100),answer)
        part.set(no_test=True)
        answer = """...++....+.....+........+.....+++............+.+.+..+..+.+..+....+..........+...+....+...+..........
++....+................+..+..+....+..+..++.+......+.....+....++...................++.......++..+....
..+........+..+.+....+......+......+......+...+............+...+..+......+....++.+.....+..+.....+.+.
.....+.+..+.++....+++.+..+.......+..+.+.........+.....+...+...............+...........+.+....+......
........+........+.........+...........+....+......+.+..........+..++++++..+.+......+.........+..+.+
"""
        self.assertEqual(plot_partition(part,100),answer)
        self.assertEqual(len(part),25)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(80,20,0))
        part.set(samplepct=40, max_reps=2)
        self.assertEqual(len(part),25)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(32,8,10))
        answer = """| |   .      ..+..... | +..        |.   | ..|+.+.+  |.| .+. +.    . . .|.   +   |. ..  .   .. .   ..
| |   +      ........ | ..+        |.   | .+|.....  |.| +.. .+    . . .|.   .   |. +.  .   ++ .   ..
| |   .      .+.+.... | ...        |.   | +.|.+...  |.| ... ..    + . .|.   .   |+ ..  +   .. .   +.
| |   .      +....+++ | .+.        |+   | ..|...+.  |.| ..+ ..    . . .|.   .   |. ..  .   .. .   ..
| |   .      ....+... | ...        |.   | ..|.....  |+| ... ..    . + +|+   .   |. .+  .   .. +   .+
. .   .      .||..||. . +.|        ..   . ...+.+.|  +|. .+. +|    | . ...   +   +. ..  .   .. |   ..
+ .   +      .||..||. . ..|        ..   + .+.....|  .|. +.. .|    | . ...   .   .. +.  .   ++ |   ..
. +   .      .||+.||. . ..|        +.   . +...+..|  .|. ... .|    | . ...   .   .+ ..  +   .. |   +.
. .   .      +||..||+ + .+|        .+   . ......+|  .|+ ..+ .|    | . ...   .   .. ..  .   .. |   ..
. .   .      .||.+||. . ..|        ..   . ..+....|  .|. ... .|    | + +++   .   .. .+  .   .. |   .+
"""
        self.assertEqual(plot_partition(part,100),answer)
        part.set( max_reps=2)
        answer = """|.|+|...|+|.|..+......|.+....|+++.||....|...|+.+.+..|.||.+.|+...|+.....|....+..||....+...+..........
|+|.|.+.|.|.|.........|+..+..|....||.+..|+.+|.....+.|.||+..|.++.|......|.......||.++.......++..+....
|.|.|...|.|+|.+.+....+|.....+|....||....|.+.|.+.....|.||...|...+|.+....|.+....+||+.....+..+.....+.+.
|.|.|+.+|.|.|+....+++.|..+...|...+||+.+.|...|...+...|.||..+|....|......|..+....||.....+.+....+......
|.|.|...|.|.|....+....|....+.|....||...+|...|......+|+||...|....|..++++|+..+.+.||...+.........+..+.+
...|+....+....||..||....+.|.|.+|+....|.......+.+.|.|+|.+.+..+|.|.+||........+...+.|..+|..+...||.....
++.|..+.......||..||...+..|.|+.|..+..|..++.+.....|+|.|..+....|+|..||..............|+..|....++||+....
..+|.......+..||+.||.+....|.|..|...+.|....+...+..|.|.|.....+.|.|..||.....+....++.+|...|+..+..||.+.+.
...|.+.+..+.++||..||+.+..+|.|..|.+..+|+.........+|.|.|+...+..|.|..||......+.......|...|.+....||.....
...|....+.....||.+||......|+|..|.....|.+....+....|.|.|.......|.|+.||+++++..+.+....|.+.|......||..+.+
"""
        self.assertEqual(plot_partition(part,100),answer)

    def test_p100f5r5h(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set(max_folds=0)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(80,0,20))
        self.assertEqual(len(part),25)
        part.set(max_folds=0,samplepct=80)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(80,0,20))
        self.assertEqual(len(part),25)
        part.set(max_folds=0,samplepct=50)
        for i in part:
            self.assertTupleEqual((len(part.T(**i)),len(part.V(**i)),len(part.S(**i))),(60,0,15))
        answer = """| |.  ..| |  .........|....   .  .||....|...|..... .|.||...|....  . . .|.  .. . |. ..... . .. . ....
. .|  ... .  .||..||......|   .  ....|...........| |.|.......|.|  | . ...  .. . .. ...|. . .. | ....
. ..  ... .  |.......|.....   |  .....|..|......|. .....||..|...  . | ...  .. . .| .|..| . |. . .|..
. ..  |.. .  .......|......   .  |..|.....||.|||.. ...........|.  . . |..  |. . .. ..|.. . .. . |..|
. ..  .|. .  ...||.....|||.   .  ......|.......... .......|.....  . . ..|  .| | .. |.... | .| . ..|.
"""
        self.assertEqual(plot_partition(part,100),answer)
        self.assertEqual(len(part),25)

    def test_max_reps0(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        answer = """...++....+.....+........+.....+++............+.+.+..+..+.+..+....+..........+...+....+...+..........
++....+................+..+..+....+..+..++.+......+.....+....++...................++.......++..+....
..+........+..+.+....+......+......+......+...+............+...+..+......+....++.+.....+..+.....+.+.
.....+.+..+.++....+++.+..+.......+..+.+.........+.....+...+...............+...........+.+....+......
........+........+.........+...........+....+......+.+..........+..++++++..+.+......+.........+..+.+
"""
        self.assertEqual(plot_partition(part,100),answer)

    def test_no_test(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        part.set( no_test=True)
        answer = """...++....+.....+........+.....+++............+.+.+..+..+.+..+....+..........+...+....+...+..........
++....+................+..+..+....+..+..++.+......+.....+....++...................++.......++..+....
..+........+..+.+....+......+......+......+...+............+...+..+......+....++.+.....+..+.....+.+.
.....+.+..+.++....+++.+..+.......+..+.+.........+.....+...+...............+...........+.+....+......
........+........+.........+...........+....+......+.+..........+..++++++..+.+......+.........+..+.+
"""
        self.assertEqual(plot_partition(part,100),answer)

    def test_no_gcv(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        part.set( no_test=True)
        part.set( no_gcv=True )
        answer = """|.|.|...|.|.|.........|......|....||....|...|.......|.||...|....|......|.......||...................
...|..........||..||......|.|..|.....|...........|.|.|.......|.|..||..............|...|......||.....
.|...........|.......|........|.......|..|......|.......||..|....|..||...........|..|..||.||.....|..
.....||....|........|...........||..|.....||.|||..|...........|.......|....|.|.......|..........|..|
.......|.|......||.....|||.|...........|..................|.............|||.|.|....|.....|..|..|..|.
"""
        self.assertEqual(plot_partition(part,100),answer)

    def test_max_folds(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        part.set( no_test=True)
        part.set( no_gcv=True )
        answer = """|.|.|...|.|.|.........|......|....||....|...|.......|.||...|....|......|.......||...................
...|..........||..||......|.|..|.....|...........|.|.|.......|.|..||..............|...|......||.....
.|...........|.......|........|.......|..|......|.......||..|....|..||...........|..|..||.||.....|..
.....||....|........|...........||..|.....||.|||..|...........|.......|....|.|.......|..........|..|
.......|.|......||.....|||.|...........|..................|.............|||.|.|....|.....|..|..|..|.
"""
        part.set( max_folds=0 )
        self.assertEqual(plot_partition(part,100),answer)

    def test_mr1_mf1(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        part.set( no_test=True)
        part.set( no_gcv=True )
        part.set( max_folds=1, max_reps=1 )
        answer = "|.|+|...|+|.|..+......|.+....|+++.||....|...|+.+.+..|.||.+.|+...|+.....|....+..||....+...+..........\n"
        self.assertEqual(plot_partition(part,100),answer)

    def test_no_test_no_gcv(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        part.set( no_test=True)
        part.set( no_gcv=True )
        part.set( max_folds=1, max_reps=1 )
        part.set( no_test=True, no_gcv=True )
        answer = "....................................................................................................\n"
        self.assertEqual(plot_partition(part,100),answer)

    def test_no_test_no_gcv_no_reps_no_folds(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        part.set( no_test=True)
        part.set( no_gcv=True )
        part.set( max_folds=1, max_reps=1 )
        part.set( no_test=True, no_gcv=True )
        answer = "....................................................................................................\n"
        part.set( no_test=True, no_gcv=True , max_reps=0, max_folds=0)
        self.assertEqual(plot_partition(part,100),answer)

    def test_no_test_no_gcv_no_reps(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        part.set( max_reps=0)
        part.set( no_test=True)
        part.set( no_gcv=True )
        part.set( max_folds=1, max_reps=1 )
        part.set( no_test=True, no_gcv=True )
        answer = "....................................................................................................\n"
        part.set( no_test=True, no_gcv=True , max_reps=0, max_folds=0)
        part.set( no_test=True, no_gcv=True , max_reps=0)
        self.assertEqual(plot_partition(part,100),answer)

    def test_get_item(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        x=part[1]
        self.assertDictEqual(x,{'k': 1, 'r': 0})

    def test_set_partitions(self):
        part = Partition(size=100,folds=5,reps=5,total_size=120)
        selection = [[0,1],[1,2]]
        part.set(partitions=selection)
        for p in part:
            self.assertIn(p, [{'r':i[0],'k':i[1]} for i in selection])

        #invalid partitions
        selection = [[0,1],[6,2]]
        with self.assertRaises(AssertionError):
            part.set(partitions=selection)

        #k=-1
        selection = [[1,-1],[2,-1]]
        part.set(partitions=selection)
        for p in part:
            self.assertIn(p, [{'r':i[0],'k':i[1]} for i in selection])

    def test_partition_T(self):
        """Legacy test to make sure refactoring in Partition matches old results. """
        Z = Partition(size=1000)
        for p in chain(Z, [{'k':-1, 'r': 0}, {'k': 1, 'r': -1},
                           {'k': -1, 'r': -1}]):
            rows = Z.T(**p)
            self.assertEqual(rows, legacy_T(Z, **p))

            out = Z.V(**p)
            self.assertEqual(out, legacy_V(Z, **p))

            out = Z.S(**p)
            self.assertEqual(out, legacy_S(Z, **p))

            out = Z.S(ignore_samplesize=True, **p)
            self.assertEqual(out, legacy_S(Z, ignore_samplesize=True, **p))

            out = Z.A(**p)
            self.assertEqual(out, legacy_A(Z, **p))


if __name__=='__main__':
    unittest.main()
