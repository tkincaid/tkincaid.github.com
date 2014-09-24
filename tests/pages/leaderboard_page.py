import time

from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys

from page import Locator
from datarobot_page import DataRobotPage
from page import retry_on_stale_reference, retry_on_timeout

class LeaderboardPage(DataRobotPage):

    LOADED_ELEMENT = Locator(By.ID, 'leaderBoard')
    SAMPLE_SIZE_INPUT = Locator(By.CLASS_NAME, 'new-sample-size')
    SAMPLE_SIZE_MENU = Locator(By.CLASS_NAME, 'samplesize-dropdown')
    SAMPLE_SIZE_SUBMIT = Locator(By.CSS_SELECTOR, '.sample-open .flatbutton')
    SELECTED_MODEL = Locator(By.CSS_SELECTOR, '.highlight')
    #MODEL_DESC = Locator(By.CLASS_NAME, 'modelDesc')


    @retry_on_stale_reference
    def select_model(self, index, timeout = None):
        """
            Wait for a model to complete

            Parameters
            ----------

            index: int
                zero-based index for the model list. For example 0 means "wait for the first"
            timeout: int
                Optional, defailts to DEFAULT_TIMEOUT

            Parameters
            ----------
            Returns the element found.
        """
        if not timeout:
            timeout = self.DEFAULT_TIMEOUT

        leaderboard_driver_wait = WebDriverWait(self.selenium, timeout)
        locator = (By.XPATH, '//div[@id][%d]' % (index + 1) )

        model = leaderboard_driver_wait.until(EC.visibility_of_element_located(locator), 'The model at position %s (zero-based) did not finish on time (%s secs)' % (index, timeout))
        model.click()
        return model.get_attribute('id')

    @retry_on_stale_reference
    @retry_on_timeout
    def submit_sample_size(self, model_data_id, sample_size):
        model = self.get_model_by_data_id(model_data_id)

        plus_button = model.find_element_by_css_selector('.samplesize .fa-plus')
        plus_button.click()

        xpath = '//div[@id="{0}"]//*[contains(@class, "{1}")]'.format(model_data_id, self.SAMPLE_SIZE_INPUT.value)
        locator = (By.XPATH, xpath)

        # Give the animation a sec, sometimes it finds the input but it can't type and throws
        # MoveTargetOutOfBoundsException: Message: Element cannot be scrolled into view
        time.sleep(0.5)

        sample_size_input = self.click(locator, wait = True, error_message = 'Could not find the sample size input. Locator: {0}'.format(xpath))
        sample_size_input.clear()
        sample_size_input.send_keys(str(sample_size))
        sample_size_input.send_keys(Keys.RETURN)

        self.wait.until(EC.invisibility_of_element_located(self.SAMPLE_SIZE_MENU))

    @retry_on_stale_reference
    def get_model_title(self, model_data_id):
        model = self.get_model_by_data_id(model_data_id)
        title = model.find_element_by_class_name('model-name-text').text
        # Remove bp
        i = title.find(' (')
        return title[:i]

    def get_model_by_data_id(self, data_id):
        return self.selenium.find_element_by_id(data_id)
