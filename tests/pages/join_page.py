from urlparse import urljoin

from selenium.webdriver.common.by import By

from datarobot_page import DataRobotPage
from page import Locator

class JoinPage(DataRobotPage):

    EMAIL = Locator(By.ID, 'email')
    PASSWORD = Locator(By.ID, 'password')
    PASSWORD_CONFIRMATION = Locator(By.ID, 'password-confirmation')
    TERMS_OF_SERVICE = Locator(By.ID, 'toscb')

    def go_to_join_page(self, base_url, username, invite_code):
        join_url = urljoin(base_url, 'join?email={0}&code={1}'.format(username, invite_code))
        print 'Joining at: {0}'.format(join_url)
        self.selenium.get(join_url)

    def sign_up(self, password):
        self.type(self.PASSWORD, password)
        self.type(self.PASSWORD_CONFIRMATION, password)

        tos_checkbox = self.click(self.TERMS_OF_SERVICE)
        tos_checkbox.submit()