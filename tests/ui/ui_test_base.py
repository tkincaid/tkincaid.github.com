import os
import time
import unittest

from pyvirtualdisplay import Display
from selenium import webdriver

from tests.pages.front_page import FrontPage

class UITestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.username =  os.environ.get('MMAPP_UI_TEST_USERNAME')
        cls.invite_code =  os.environ.get('MMAPP_UI_TEST_INVITE_CODE')
        cls.password =  os.environ.get('MMAPP_UI_TEST_PASSWORD')
        cls.screenshots_path =  os.environ.get('MMAPP_UI_SCREENSHOTS_PATH')

        if not cls.username or not cls.invite_code or not cls.password:
            raise ValueError('Username, invite code and password are required. Set the respective env vars')

        if os.environ.get('MMAPP_UI_TEST_DISABLE_GUI') == '1':
            cls.display = Display(visible = 0, size = (1600, 1200))
            cls.display.start()
        else:
            cls.display = None

        cls.base_url = os.environ.get('MMAPP_UI_TEST_URL')
        if not cls.base_url:
            cls.base_url = 'http://user:pass@localhost'

        # Create folder for screenshots
        if cls.screenshots_path and not os.path.isdir(cls.screenshots_path):
            os.makedirs(cls.screenshots_path)

        cls.browser = webdriver.Firefox()
        cls.frontpage = FrontPage(cls.browser)

    def take_screenshot(self, file_name):
        self.browser.save_screenshot(os.path.join(self.screenshots_path, file_name))

    def tearDown(self):
        if self.screenshots_path:
            self.take_screenshot(self.id() + '.png')

    @classmethod
    def tearDownClass(cls):
        if cls.frontpage:
            try:
                cls.frontpage.log_out()
            except Exception, e:
                print('Could not logout: {0}'.format(e))

        cls.browser.quit()

        if cls.display:
            cls.display.stop()
