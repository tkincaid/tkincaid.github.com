import time
import unittest

from tests.ui.ui_test_base import UITestBase
from tests.pages.data_page import DataPage

class UITestEDA(UITestBase):

    def test_eda(self):
        # Login to the app
        # Assumes that a dataset was uploaded in a previous test
        self.frontpage.login(self.base_url, self.username, self.password)

        # Check that the page loaded properly
        datapage = DataPage(self.browser)
        self.assertTrue(datapage.loaded)

        # Click on a feature and scroll through the charts
        # TODO: Assert that everything was rendered properly
        column = datapage.click_column_index(2)
        datapage.single_var_go_to_next(column)
        datapage.single_var_go_to_next(column)
        datapage.single_var_go_to_next(column)

if __name__ == '__main__':
    unittest.main()
