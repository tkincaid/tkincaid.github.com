from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys

from page import Locator
from datarobot_page import DataRobotPage

class AimPage(DataRobotPage):

    AIM_POPUP = Locator(By.CLASS_NAME, 'aim_content')
    TARGET_VARIABLE_INPUT = Locator(By.ID, 'target-feature-select')
    RANKING_METRIC_BUTTON = Locator(By.CSS_SELECTOR, '.metric-box .aim-button-clicked')
    AUTO_BUTTON = Locator(By.CLASS_NAME, 'automatically')
    SEMI_AUTO_BUTTON = Locator(By.CLASS_NAME, 'semi-automatically')
    MANUAL_BUTTON = Locator(By.CLASS_NAME, 'manually')
    START_BUTTON = Locator(By.CSS_SELECTOR, '.start-button h1.dogo')

    def __init__(self, selenium):
        super(AimPage, self).__init__(selenium)

    def click_start_button(self):
        self.click(self.START_BUTTON)

    def is_visible(self):
        # TODO: Create a generalized version of this method in page.py
        try:
            wait = WebDriverWait(self.selenium, 10)
            wait.until(EC.visibility_of_element_located((By.CLASS_NAME, self.AIM_POPUP.value)))

            return True
        except TimeoutException:
            # TODO: Include more debugging output
            return False

    def select_target_variable(self, variable):
        target_input = self.type(self.TARGET_VARIABLE_INPUT, variable, error_message = "Could not type the target variable")
        target_input.send_keys(Keys.RETURN)
        return True

    def select_metric(self, metric):
        if metric == 'Ranking':
            self.click(self.RANKING_METRIC_BUTTON)

    def set_modeling_mode(self, mode):
        """
        Sets the initial modeling mode.

        Parameters
        ----------
        mode : str
            The valid values for this are 'auto', 'semi-auto', and 'manual'
        """
        self.click(mode)

    def explore_data(self):
        """Chooses to explore the data first instead of selecting a modeling mode"""
        self.click(self.EXPLORE_DATA_BUTTON)

