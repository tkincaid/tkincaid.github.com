from selenium.webdriver.common.by import By

from page import Page, Locator, retry_on_stale_reference, retry_on_timeout

class DataRobotPage(Page):

    LOG_OUT_LINK = Locator(By.ID, 'user_logout_link')
    SIGN_UP_LINK = Locator(By.ID, 'user_signup_link')
    USERNAME_INPUT = Locator(By.ID, 'username')
    PASSWORD_INPUT = Locator(By.ID, 'password')


    DATA_TAB = Locator(By.CSS_SELECTOR, '#nav ul > li:nth-child(1)')
    MODELS_TAB = Locator(By.CSS_SELECTOR, '#nav li:nth-child(2) a')
    INSIGHTS_TAB = Locator(By.CSS_SELECTOR, '#nav ul > li:nth-child(3)')
    RSTUDIO_TAB = Locator(By.CSS_SELECTOR, '#nav ul > li:nth-child(4)')
    PYTHON_TAB = Locator(By.CSS_SELECTOR, '#nav ul > li:nth-child(5')
    TASKS_TAB = Locator(By.CSS_SELECTOR, '#nav ul > li:nth-child(6)')

    PROJECT_VIEW = Locator(By.ID, 'new-project')

    def __init__(self, selenium):
        super(DataRobotPage, self).__init__(selenium)

    def log_out(self):
        self.click(self.LOG_OUT_LINK, error_message='Could not click logout link')

    def sign_up(self, username, password):
        self.click(self.SIGN_UP_LINK, error_message='Could not click sign up link')
        self.type(self.USERNAME_INPUT, username)
        self.type(self.PASSWORD_INPUT, password)

    def select_models_tab(self):
        self.click(self.MODELS_TAB)

    def select_data_tab(self):
        self.click(self.DATA_TAB)

    @retry_on_stale_reference
    def is_in_upload_page(self):
        self.click(self.PROJECT_VIEW, wait = True, error_message = 'The front page did not load on time')
