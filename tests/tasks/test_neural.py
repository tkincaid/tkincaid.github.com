"""Unit tests for the neural network module"""
import unittest

import numpy as np

import ModelingMachine.engine.tasks.neural as neural
import ModelingMachine.engine.tasks.neural_tasks as neural_tasks

###############################################################
# Tests of the underlying Neural Network code
#


class TestNNTraining(unittest.TestCase):

    def test_init(self):
        batchsize = 50
        maxepoch = 125
        mu = 0.9
        eta = 0.1
        trainer = neural.NNTrainer(batchsize, maxepoch, mu, eta)


class TestNeuralNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.arch = [2, 10, 1]
        cls.regs = [neural.LayerReg() for i in range(2)]
        cls.loss_type = neural.LossTypes.MSE
        cls.rng = np.random.RandomState(0)
        cls.data = cls.rng.randn(100, 2)

    def test_init_logistic(self):
        layer_types = [neural.LayerTypes.LOGISTIC] * 2
        net = neural.NeuralNet(self.arch, layer_types, self.regs,
                               self.loss_type)
        values = net.predict(self.data)
        self.assertEqual(len(values.shape), 1)

    def test_init_linear(self):
        layer_types = [neural.LayerTypes.LINEAR] * 2
        net = neural.NeuralNet(self.arch, layer_types, self.regs,
                               self.loss_type)
        values = net.predict(self.data)
        self.assertEqual(len(values.shape), 1)

    def test_init_rectified(self):
        layer_types = [neural.LayerTypes.RECTIFIED] * 2
        net = neural.NeuralNet(self.arch, layer_types, self.regs,
                               self.loss_type)
        values = net.predict(self.data)
        self.assertEqual(len(values.shape), 1)

    def test_init_sans_loss_type(self):
        layer_types = [neural.LayerTypes.RECTIFIED] * 2
        net = neural.NeuralNet(self.arch, layer_types, self.regs)
        values = net.predict(self.data)
        self.assertEqual(len(values.shape), 1)

    def test_unknown_raises_error(self):
        self.assertRaises(
            ValueError, neural.NNLayer.create, 1337, 10, 20, neural.LayerReg())


class TestNeuralNetTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        regs = [neural.LayerReg() for i in range(2)]
        regs[1].dropout = False
        cls.net = neural.NeuralNet([4, 10, 1],
                                   [neural.LayerTypes.LOGISTIC,
                                    neural.LayerTypes.LINEAR],
                                   regs,
                                   neural.LossTypes.MSE)
        cls.random_state = np.random.RandomState(0)
        cls.data = cls.random_state.randn(125, 4)
        cls.target = np.abs(cls.data[:, 0]) + cls.data[:, 1]**2 - \
            cls.data[:, 3]
        cls.target = cls.target.reshape(-1, 1)

    def test_training(self):
        trainer = neural.NNTrainer(batchsize=20, maxepoch=10, momentum=0.5,
                                   learn_rate=0.1)
        trainer.train_sgd(self.net, self.data, self.target)

    def test_training_with_custom_scalar_schedules(self):
        momentum = neural.ExponentialSchedule(1.0, 0.98)
        learn_rate = neural.LinearSchedule(0.5, 0.01, 10)
        trainer = neural.NNTrainer(batchsize=20, maxepoch=10,
                                   momentum=momentum,
                                   learn_rate=learn_rate)
        trainer.train_sgd(self.net, self.data, self.target)

    def test_training_with_single_dim_target(self):
        momentum = neural.ExponentialSchedule(1.0, 0.98)
        learn_rate = neural.LinearSchedule(0.5, 0.01, 10)
        trainer = neural.NNTrainer(batchsize=20, maxepoch=10,
                                   momentum=momentum,
                                   learn_rate=learn_rate)
        trainer.train_sgd(self.net, self.data, self.target.flatten())

    def test_training_with_early_stop_heuristic(self):
        def stop_heuristic(errors):
            return len(errors) >= 10

        trainer = neural.NNTrainer(batchsize=20, maxepoch=50, momentum=0.5,
                                   learn_rate=0.1)
        errs = trainer.train_sgd(
            self.net, self.data, self.target, stop_heuristic)

        self.assertEqual(len(errs), 10)


class TestStoppingHeuristic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.RandomState(0)
        cls.errs = cls.rng.randn(100)*np.linspace(0.0, 0.1, 100) + \
            np.append(np.linspace(5.0, 4.0, 80), [4.0]*20)
        cls.heur = neural.TrendStopHeuristic(window=10, patience=10)

    def test_does_at_least_10(self):
        self.assertFalse(self.heur(self.errs[:9]))

    def test_obviously_should_stop(self):
        self.assertTrue(self.heur(self.errs))


class TestNetFit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        regs = [neural.LayerReg() for i in range(2)]
        regs[1].dropout = False
        cls.net = neural.NeuralNet([4, 10, 1],
                                   [neural.LayerTypes.LOGISTIC,
                                    neural.LayerTypes.LINEAR],
                                    regs,
                                    neural.LossTypes.MSE)
        cls.random_state = np.random.RandomState(0)
        cls.data = cls.random_state.randn(125, 4)
        cls.target = np.abs(cls.data[:, 0]) + cls.data[:, 1]**2 - \
            cls.data[:, 3]
        cls.target = cls.target.reshape(-1, 1)

    def test_training(self):
        self.net.fit(self.data, self.target)

    def test_training_with_custom_scalar_schedules(self):
        self.net.fit(self.data, self.target,
                     momentum_update=neural.ScheduleTypes.EXPONENTIAL,
                     learn_rate_update=neural.ScheduleTypes.LINEAR)


class TestSchedules(unittest.TestCase):

    def test_linear_schedule(self):
        sched = neural.LinearSchedule(0.0, 10.0, 10)
        for i in range(10):
            x = sched.get()
            self.assertEqual(x, i)

    def test_exp_schedule(self):
        sched = neural.ExponentialSchedule(1, 1.01)
        comp = 1.0
        for i in range(10):
            x = sched.get()
            self.assertEqual(comp, x)
            comp *= 1.01

    def test_constant_schedule(self):
        sched = neural.ConstantSchedule(0.5)
        for i in range(10):
            self.assertEqual(sched.get(), 0.5)


class TestBatch(unittest.TestCase):

    def test_with_rand_uniqueness(self):
        nsamples = 125
        batch_size = 10
        batches = neural.get_batch_indices(nsamples, batch_size, rand_seed=25)

        seen = set()
        for batch in batches:
            self.assertTrue(np.all(batch < nsamples))
            for idx in batch:
                self.assertNotIn(idx, seen)
                seen.add(idx)

class TestNumericalAccuracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nsamples = 1
        cls.n_in_dims = 5
        cls.n_out_nodes = 1

    def setUp(self):
        self.rng = np.random.RandomState(1)
        self.X = self.rng.randn(self.nsamples, self.n_in_dims)
        self.Y = self.rng.choice([0, 1], [self.nsamples, 1])


    def test_simple_linear_with_MSE(self):
        arch = [5, 1]
        lts = [neural.LayerTypes.LINEAR]
        regs = [neural.LayerReg(weight_penalty=0.0, dropout=False)]
        loss = neural.LossTypes.MSE
        net = neural.NeuralNet(arch, lts, regs, loss_type=loss)

        num_grad = numerical_gradient(net, self.X, self.Y)
        calc_grad = neural.model_gradient(net, self.X, self.Y)

        np.testing.assert_almost_equal(num_grad, calc_grad)

    def test_simple_logistic_with_cross_entropy(self):
        arch = [5, 1]
        lts = [neural.LayerTypes.LOGISTIC]
        regs = [neural.LayerReg(weight_penalty=0.0, dropout=False)]
        loss = neural.LossTypes.CROSS_ENTROPY
        net = neural.NeuralNet(arch, lts, regs, loss_type=loss)

        num_grad = numerical_gradient(net, self.X, self.Y)
        calc_grad = neural.model_gradient(net, self.X, self.Y)

        np.testing.assert_almost_equal(num_grad, calc_grad)

    def test_combined_case(self):
        arch = [5, 6, 1]
        lts = [neural.LayerTypes.LOGISTIC, neural.LayerTypes.LINEAR]
        regs = [neural.LayerReg(weight_penalty=0., dropout=False)
                for lt in lts]
        loss = neural.LossTypes.MSE
        net = neural.NeuralNet(arch, lts, regs, loss_type=loss)

        num_grad = numerical_gradient(net, self.X, self.Y)
        calc_grad = neural.model_gradient(net, self.X, self.Y)

        np.testing.assert_almost_equal(num_grad, calc_grad)


def numerical_gradient(net, data, targets):
    theta_0 = net.get_theta()
    epsilon = 1e-5
    num_grad = np.zeros_like(theta_0)
    for idx in range(len(theta_0)):
        eps_vect = np.zeros_like(theta_0)
        eps_vect[idx] = epsilon
        theta_plus = theta_0 + eps_vect
        theta_minus = theta_0 - eps_vect

        net.set_theta(theta_plus)
        cost_plus = neural.model_cost(net, data, targets, net.loss_type)[1]

        net.set_theta(theta_minus)
        cost_minus = neural.model_cost(net, data, targets, net.loss_type)[1]
        num_grad[idx] = ((cost_plus - cost_minus) / (2 * epsilon)).flatten()[0]
        net.set_theta(theta_0)
    return num_grad


class TestModelerConstructors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(2)
        cls.X = rng.randn(10, 8)
        cls.Y = rng.choice([0, 1], (10, 1))

    def test_nn1c(self):
        model = neural_tasks._ShallowNetClassifier(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)

    def test_nn1r(self):
        model = neural_tasks._ShallowNetRegressor(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)

    def test_vnnc(self):
        model = neural_tasks._VanillaNetClassifier(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)

    def test_vnnr(self):
        model = neural_tasks._VanillaNetRegressor(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)

    def test_vnn2c(self):
        model = neural_tasks._VanillaTwoLayerNetClassifier(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)

    def test_vnn2r(self):
        model = neural_tasks._VanillaTwoLayerNetRegressor(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)

    def test_nn2c(self):
        model = neural_tasks._DrednetClassifier(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)

    def test_nn2r(self):
        model = neural_tasks._DrednetRegressor(max_epoch=1)
        self.assertRaises(RuntimeError, model.predict, self.X)
        model.fit(self.X, self.Y)
