
import unittest

from ModelingMachine.engine.workrequest import WorkRequest, mongodict

class TestWorkRequest(unittest.TestCase):

    def test_mongodict(self):
        x = {'a':1,'b':{'a':1,'c':2},'c':[1,2],'d':'asdf'}
        out = mongodict(x)
        self.assertEqual(out, {'a': 1, 'b.c': 2, 'c': [1, 2], 'b.a': 1, 'd': 'asdf'})

        x = {'a':1,'b':{'a':1,'f':{'g':1,'h':2}},'c':[1,2],'d':'asdf'}
        out = mongodict(x)
        self.assertEqual(out, {'a': 1, 'c': [1, 2], 'd': 'asdf', 'b.f.h': 2, 'b.a': 1, 'b.f.g': 1})

        x = {'partition_stats':{str((0,-1)):{'a':1}},'a':{'b':1}}
        out = mongodict(x)
        self.assertEqual(out, {'partition_stats.'+str((0,-1)):{'a':1},'a.b':1})

    def test_part_key(self):
        parts = {'partitions':[[3,-1]]}
        part_reps = max([i[0]+1 for i in parts.get('partitions',[[parts.get('max_reps',0)-1,-1]])])
        self.assertEqual(part_reps,4)

        parts = {'partitions':[[i,-1] for i in range(5)]}
        part_reps = max([i[0]+1 for i in parts.get('partitions',[[parts.get('max_reps',0)-1,-1]])])
        self.assertEqual(part_reps,5)

        parts = {'max_reps':1}
        part_reps = max([i[0]+1 for i in parts.get('partitions',[[parts.get('max_reps',0)-1,-1]])])
        self.assertEqual(part_reps,1)

        parts = {'max_reps':5}
        part_reps = max([i[0]+1 for i in parts.get('partitions',[[parts.get('max_reps',0)-1,-1]])])
        self.assertEqual(part_reps,5)

        parts = {'partitions':[[3,-1]],'max_reps':1}
        part_reps = max([i[0]+1 for i in parts.get('partitions',[[parts.get('max_reps',0)-1,-1]])])
        self.assertEqual(part_reps,4)

if __name__ == '__main__':
    unittest.main()
