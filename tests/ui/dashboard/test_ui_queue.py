import time
import unittest

from tests.ui.ui_test_base import UITestBase
from tests.pages.queue_page import QueuePage

class UITestQueue(UITestBase):

    def test_queue(self):
        self.frontpage.login(self.base_url, self.username, self.password)

        queue = QueuePage(self.browser, timeout = 20)
        # Check that active and processing are not empty
        self.assertTrue(len(queue.get_processing_models()) != 0)
        self.assertTrue(len(queue.get_waiting_models()) != 0)

        # Check to see if the resources are updating (big timeout, with small datasets some go so quick the screen doesn't update)
        queue.check_resource_usage()

        # Remove items from the queue
        # Add a .1 to .2 delay between removals to make it more realistic
        # for model in queue.get_waiting_models():
        #     queue.remove(model)
        #     time.sleep(.1)

        # Wait for the queue to be empty
        # Check for the autopilot popup
        # Choose determine next steps
        # Kickoff a leaderboard item
        # Check for autopilot
if __name__ == '__main__':
    unittest.main()
