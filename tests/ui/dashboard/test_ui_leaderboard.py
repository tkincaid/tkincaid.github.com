import unittest

from tests.ui.ui_test_base import UITestBase
from tests.pages.leaderboard_page import LeaderboardPage
from tests.pages.queue_page import QueuePage


class UITestLeaderboard(UITestBase):

    def test_leaderboard(self):
        self.frontpage.login(self.base_url, self.username, self.password)
        self.frontpage.select_models_tab()

        leaderboard_page = LeaderboardPage(self.browser)
        # ACTION:   Click on the Modeling menu item after at least one item is complete
        # EXPECTED: The leaderboard displays
        model_data_id = leaderboard_page.select_model(index = 0, timeout = 20)
        model_title = leaderboard_page.get_model_title(model_data_id)

        # ACTION:   Click on the plus icon under the Training Sample Size column of the first item on the board
        # EXPECTED: A small popup appears showing a percent value

        # ACTION:   Change to x% and click Run
        # EXPECTED: The popup closes
        # EXPECTED: A new item is added to the queue
        new_sample_size = 81
        leaderboard_page.submit_sample_size(model_data_id, new_sample_size)

        queue = QueuePage(self.browser)
        item = queue.get_item(model_title, new_sample_size)
        self.assertIsNotNone(item)

        # ACTION:   Click on Single Model
        # EXPECTED: Lift Chart is selected
        # EXPECTED: A graph is displayed
        # ACTION:   Click ROC curve
        # EXPECTED: A graph is displayed
        # ACTION:   Click Model Info
        # EXPECTED: The following information is displayed: Blueprint, Tuning Parameters, Sample Size, Run Time

if __name__ == '__main__':
    unittest.main()
