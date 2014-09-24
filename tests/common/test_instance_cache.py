import threading
import unittest

from common.utilities.instance_cache import CachedInstantiator

JOIN_TIMEOUT = 1


class ExampleInstanceCache(CachedInstantiator):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class ExampleThreadedInstanceCache(CachedInstantiator):
    protect_thread = True

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


class TestInstanceCache(unittest.TestCase):
    def setUp(self):
        ExampleInstanceCache._instance_list = []
        ExampleThreadedInstanceCache._instance_list = []

    def test_same_args(self):
        obj1 = ExampleInstanceCache.instance(1, 2, 3)
        obj2 = ExampleInstanceCache.instance(1, 2, 3)
        self.assertIs(obj1, obj2)

    def test_different_args(self):
        obj1 = ExampleInstanceCache.instance(1, 2, 3)
        obj2 = ExampleInstanceCache.instance(1, 2, 5)
        self.assertIsNot(obj1, obj2)

    def test_properly_instantiates(self):
        obj1 = ExampleInstanceCache.instance(1, 2, 3)
        self.assertEqual(obj1.a, 1)
        self.assertEqual(obj1.b, 2)
        self.assertEqual(obj1.c, 3)

    def test_destroy(self):
        obj1 = ExampleInstanceCache.instance(1, 2, 3)
        obj1_id = id(obj1)
        obj1.destroy()
        obj2 = ExampleInstanceCache.instance(1, 2, 3)
        self.assertNotEqual(obj1_id, id(obj2))

    def test_normal_instantiate(self):
        obj1 = ExampleInstanceCache.instance(1, 2, 3)
        obj2 = ExampleInstanceCache(1, 2, 3)
        obj3 = ExampleInstanceCache.instance(1, 2, 3)
        self.assertIsNot(obj1, obj2)
        self.assertIsNot(obj2, obj3)
        self.assertIs(obj1, obj3)

    def test_protected_thread_different(self):
        ids = []

        def create_and_save():
            obj1 = ExampleThreadedInstanceCache.instance(1, 2, 3)
            ids.append(obj1)
        t1 = threading.Thread(target=create_and_save)
        t2 = threading.Thread(target=create_and_save)
        t1.start()
        t2.start()
        t1.join(JOIN_TIMEOUT)
        t2.join(JOIN_TIMEOUT)
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])

    def test_unprotected_thread_same(self):
        ids = []

        def create_and_save():
            obj1 = ExampleInstanceCache.instance(1, 2, 3)
            ids.append(obj1)
        t1 = threading.Thread(target=create_and_save)
        t2 = threading.Thread(target=create_and_save)
        t1.start()
        t2.start()
        t1.join(JOIN_TIMEOUT)
        t2.join(JOIN_TIMEOUT)
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], ids[1])

    def test_cleanup_stale_threads(self):
        def create_and_save():
            ExampleThreadedInstanceCache.instance(1, 2, 3)
        t1 = threading.Thread(target=create_and_save)
        t2 = threading.Thread(target=create_and_save)
        t1.start()
        t2.start()
        t1.join(JOIN_TIMEOUT)
        t2.join(JOIN_TIMEOUT)
        ExampleThreadedInstanceCache.instance(1, 2, 3)
        self.assertEqual(len(ExampleThreadedInstanceCache._instance_list), 1)

if __name__ == '__main__':
    unittest.main()
