from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

from page import Locator, retry_on_stale_reference, retry_on_timeout
from datarobot_page import DataRobotPage

class QueuePage(DataRobotPage):

    PROCESSING_MODELS = Locator(By.CSS_SELECTOR, '.top_monitor div.QueueItem')
    WAITING_MODELS = Locator(By.CSS_SELECTOR, '.queue_monitor div.QueueItem')
    QUEUE_ITEM = Locator(By.CLASS_NAME, 'QueueItem')
    QUEUE_ITEM_MODEL_NAME = Locator(By.CLASS_NAME, 'modelname')
    QUEUE_ITEM_RUNS = Locator(By.CLASS_NAME, 'runs')
    RESOURCE_USAGE_PROCESSING_GRAPHS = Locator(By.CSS_SELECTOR, '.processing_usage svg')
    RESOURCE_USAGE_MEMORY_GRAPHS = Locator(By.CSS_SELECTOR, '.memory_usage .memory_gradient_container')
    SIDEBAR = Locator(By.CLASS_NAME, 'sidebar')
    MINUS_WORKER = Locator(By.ID, 'parminus')
    WORKER_COUNT = Locator(By.CSS_SELECTOR, '.QueueSettings .parvalue')

    def __init__(self, selenium, timeout = None):
        super(QueuePage, self).__init__(selenium)

        if not timeout:
            timeout = self.DEFAULT_TIMEOUT

        self.queue_driver_wait = WebDriverWait(self.selenium, timeout)
        error_message = 'Could not locate the sidebar'
        self.side_bar = self.queue_driver_wait.until(EC.visibility_of_element_located(self.SIDEBAR), message = error_message)

    def get_processing_models(self, timeout = None):
        return self.queue_driver_wait.until(EC.presence_of_all_elements_located(self.PROCESSING_MODELS), message = 'Could not retrieve items from the in-process queue')

    @retry_on_timeout
    @retry_on_stale_reference
    def get_item(self, model_title, sample_size):

        xpath = '//div[contains(@class, "{0}")]'.format(self.QUEUE_ITEM.value) # Grab all queue items
        xpath += '//p[contains(@class, "{0}")][contains(@title,"{1}")]'.format(self.QUEUE_ITEM_MODEL_NAME.value, model_title) # Filter by model name
        xpath += '|' # and
        xpath += 'p[contains(@class, "{0}")][text()[contains(.,"{1}%")]]'.format(self.QUEUE_ITEM_RUNS.value, sample_size) # filter by sample size

        locator = (By.XPATH, xpath)
        error_message = 'Could not locate queue item: {0}. Locator: {1}'.format(model_title, xpath)

        # An expectation for checking that an element is present on the DOM of a page
        return self.queue_driver_wait.until(EC.presence_of_element_located(locator), message = error_message)

    def get_waiting_models(self):
        return self.queue_driver_wait.until(EC.presence_of_all_elements_located(self.WAITING_MODELS), message = 'Could not retrieve waiting items from the queue')

    def check_resource_usage(self, timeout = None):
        self.queue_driver_wait.until(EC.presence_of_all_elements_located(self.RESOURCE_USAGE_PROCESSING_GRAPHS), 'Processing graphs failed to display')
        self.queue_driver_wait.until(EC.presence_of_all_elements_located(self.RESOURCE_USAGE_MEMORY_GRAPHS), 'Memory graphs failed to display')

    def remove(self, model):
        pass

    def get_worker_count(self):
        workers_label = self.click(self.WORKER_COUNT, error_message = 'Could not get the worker count')
        return int(workers_label.text)

    def decrease_worker(self):
        self.click(self.MINUS_WORKER, error_message = 'Could not click minus worker')
