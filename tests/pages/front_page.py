from selenium.webdriver.common.by import By

from page import Locator
from datarobot_page import DataRobotPage

class FrontPage(DataRobotPage):

    LOADED_ELEMENT = Locator(By.ID, 'home-view')
    IMPORT_DATA_BUTTON = Locator(By.CLASS_NAME, 'openImportButton')
    UPLOAD_URL_INPUT = Locator(By.ID, 'import_url')
    UPLOAD_URL_BUTTON = Locator(By.ID, 'remoteurl_btn')
    LOGIN_BUTTON = Locator(By.ID, 'login-button')

    def __init__(self, selenium):
        super(FrontPage, self).__init__(selenium)

    def upload_url(self, url):
        #Not needed in private mode
        #self.click(self.IMPORT_DATA_BUTTON, error_message='Could not click import button')
        self.type(self.UPLOAD_URL_INPUT, url, error_message = 'Could not type url')
        self.click(self.UPLOAD_URL_BUTTON, error_message='Could not click upload button')

    def login(self, login_url, username, password):
        self.selenium.get(login_url)

        self.type(self.USERNAME_INPUT, username)
        self.type(self.PASSWORD_INPUT, password)
        self.click(self.LOGIN_BUTTON)
