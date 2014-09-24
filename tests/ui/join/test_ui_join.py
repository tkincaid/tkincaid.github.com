import unittest
import os

from tests.ui.ui_test_base import UITestBase
from tests.pages.join_page import JoinPage

class UITestJoin(UITestBase):

    def test_join(self):
        join_page = JoinPage(self.browser)
        join_page.go_to_join_page(self.base_url, self.username, self.invite_code)
        join_page.sign_up(self.password)
        join_page.is_in_upload_page()

