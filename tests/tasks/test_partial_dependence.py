#########################################################
#
#       Unit Test for tasks/gbm.py
#
#       Author: Tom de Godoy, Peter Prettenhofer
#
#       Copyright DataRobot, Inc. 2013
#
########################################################
import numpy as np
import json

from tesla.ensemble import RandomForestRegressor
from tesla.ensemble import GradientBoostingRegressor

from base_task_test import BaseTaskTest

from ModelingMachine.engine.tasks import partial_dependence
from ModelingMachine.engine.tasks.gbm import ESGBC
from ModelingMachine.engine.tasks.cat_encoders import CategoricalToOrdinalConverter
from ModelingMachine.engine.container import Container


class TestPartialDependenceGBRT(BaseTaskTest):

    def test_partial_dependence_reg(self):
        """Smoke test for regression. """
        X, Y, Z = self.create_reg_syn_data()
        task_desc = 'ESGBR md=3;n=2'
        t = self.check_task(task_desc, X, Y, Z)
        X = X.dataframe

        mdl = t.model.values()[0]

        colnames = X.columns.tolist()
        k = 2
        pdps = partial_dependence.auto_partial_dependence(mdl, X.values, Y,
                                                          colnames, [0 for c in colnames],
                                                          {}, k=k)

        # test if output format is as expected
        self.assertEqual(len(pdps), 2)
        most_important_fx = mdl.feature_importances_.argsort()[::-1][:k]
        self.assertEqual([pdp.colnames[0] for pdp in pdps],
                         [colnames[i] for i in most_important_fx])
        self.assertEqual([pdp.colindices[0] for pdp in pdps],
                         [i for i in most_important_fx])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.colnames) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.coltypes) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.colindices) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.axes) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.values) for pdp in pdps])
        self.assertEqual([100 for pdp in pdps],
                         [len(pdp.values[0]) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.freqs) for pdp in pdps])
        self.assertEqual([100 for pdp in pdps],
                         [len(pdp.freqs[0]) for pdp in pdps])

    def test_partial_dependence_clf(self):
        """Smoke test for clf. """
        X, Y, Z = self.create_reg_syn_data()
        Y = Y > np.median(Y)

        task_desc = 'ESGBC md=[3,4];n=2'
        t = self.check_task(task_desc, X, Y, Z)
        X = X.dataframe

        mdl = t.model.values()[0]

        colnames = X.columns.tolist()
        k = 2
        pdps = partial_dependence.auto_partial_dependence(mdl, X.values, Y,
                                                          colnames, [0 for c in colnames],
                                                          {}, k=k)
        self.assertEqual(len(pdps), 2)

        most_important_fx = mdl.feature_importances_.argsort()[::-1][:k]
        self.assertEqual([pdp.colnames[0] for pdp in pdps],
                         [colnames[i] for i in most_important_fx])
        self.assertEqual([pdp.colindices[0] for pdp in pdps],
                         [i for i in most_important_fx])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.colnames) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.coltypes) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.colindices) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.axes) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.values) for pdp in pdps])
        self.assertEqual([100 for pdp in pdps],
                         [len(pdp.values[0]) for pdp in pdps])
        self.assertEqual([1 for pdp in pdps],
                         [len(pdp.freqs) for pdp in pdps])
        self.assertEqual([100 for pdp in pdps],
                         [len(pdp.freqs[0]) for pdp in pdps])

        # test (de)serialization
        pdp_ser = json.dumps(pdps[0]._asdict())
        pdp_deser = partial_dependence.PartialDependencePlot(**json.loads(pdp_ser))
        self.assertEqual(pdp_deser, pdps[0])

    def test_invalid_input(self):
        """Check if invalid inputs are properly handled. """
        mdl = RandomForestRegressor()
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        coltypes = [1 for _ in range(2)]
        colnames = map(unicode, range(2))
        self.assertRaises(ValueError, partial_dependence.auto_partial_dependence,
                          mdl, X, y, colnames, coltypes, {}, k=2)

        mdl = GradientBoostingRegressor()
        coltypes = [1 for _ in range(3)]  # one too much
        self.assertRaises(ValueError, partial_dependence.auto_partial_dependence,
                          mdl, X, y, colnames, coltypes, {}, k=2)

        coltypes = [1 for _ in range(2)]
        colnames = map(unicode, range(1))  # one too few
        self.assertRaises(ValueError, partial_dependence.auto_partial_dependence,
                          mdl, X, y, colnames, coltypes, {}, k=2)

        coltypes = [1 for _ in range(2)]
        colnames = map(unicode, range(2))
        # mdl not fitted!
        self.assertRaises(ValueError, partial_dependence.auto_partial_dependence,
                          mdl, X, y, colnames, coltypes, {}, k=2)

    def test_ord_enc_metadata(self):
        """Smoke test for clf. """
        X, Y, Z = self.create_reg_data(categoricals=True)
        X = X.dataframe
        cat_cols = [c for c in X.columns if X[c].dtype not in (np.int64, np.float64)]

        X = X[cat_cols]
        ord_enc = CategoricalToOrdinalConverter()
        ord_enc.fit(Container(X), Y, Z)
        C = ord_enc.transform(Container(X), Y, Z)
        metadata = C.metadata()
        for col in X.columns:
            self.assertTrue(col in metadata)
            self.assertEqual(ord_enc.ord_enc_.col_categoricals_[col].shape[0] + 2,
                             len(metadata[col]['levels']))
        t = ESGBC('n=2')
        t.fit(C, Y, Z)
        pdps = t.partial_dependence_plots.values()[0]
        self.assertEqual(len(pdps), 10)
        pdp = pdps[0]
        self.assertEqual(pdp['axes'], [['C', 'F', 'E', 'A', 'B', '==other==']])
