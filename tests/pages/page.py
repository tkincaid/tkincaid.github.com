import time

from collections import namedtuple

from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

Locator = namedtuple('Locator', ['type', 'value'])

def retry_on_stale_reference(call):
    def _retry_on_stale_reference(*args, **kwargs):
        retry = 3
        for i in xrange(retry):
            try:
                return call(*args, **kwargs)
            except StaleElementReferenceException, e:
                if i < retry:
                    seconds = pow(2, i)
                    print('Received stale reference (atempting again in {0} seconds): {1}'.format(seconds,e))
                    time.sleep(seconds)
                else:
                    raise e
    return _retry_on_stale_reference

def retry_on_timeout(call):
    def _retry_on_timeout(*args, **kwargs):
        retry = 2
        for i in xrange(retry):
            try:
                return call(*args, **kwargs)
            except TimeoutException, e:
                if i < retry:
                    seconds = pow(2, i)
                    print('Timeout (atempting again in {0} seconds): {1}'.format(seconds,e))
                    time.sleep(seconds)
                else:
                    raise e
    return _retry_on_timeout

class Page(object):
    """
    Base page object

    This implements the page object pattern for UI testing.
    See http://code.google.com/p/selenium/wiki/PageObjects for more details
    """

    DEFAULT_TIMEOUT = 5
    LOADED_ELEMENT = None

    def __init__(self, selenium):
        self.selenium = selenium
        self.wait = WebDriverWait(selenium, self.DEFAULT_TIMEOUT)

    def click(self, locator, wait=True, timeout=None, error_message=None):
        if wait:
            if timeout:
                wait = WebDriverWait(self.selenium, timeout)
                element = wait.until(EC.element_to_be_clickable(locator), message = error_message)
            else:
                # Use the default timeout
                element = self.wait.until(EC.element_to_be_clickable(locator), message = error_message)

        else:
            # Don't wait when searching
            if locator.type == By.ID:
                element = self.selenium.find_element_by_id(locator.value)
            elif locator.type == By.CLASS_NAME:
                element = self.selenium.find_element_by_class_name(locator.value)
            elif locator.type == By.CSS_SELECTOR:
                element = self.selenium.find_element_by_css_selector(locator.value)
            elif locator.type == By.XPATH:
                element = self.selenium.find_element_by_xpath(locator.value)
            else:
                raise TypeError("Unknown locator type: {0}".format(locator.type))

        element.click()
        return element

    @retry_on_timeout
    def verify_is_loaded(self, timeout=None, error_message=None):
        if self.LOADED_ELEMENT:
            wait = self.wait
            if timeout:
                wait = WebDriverWait(self.selenium, timeout)
            wait.until(EC.visibility_of_element_located(self.LOADED_ELEMENT), message=error_message)
        else:
            raise NotImplementedError

    def type(self, locator, string, wait=True, timeout=None, error_message=None):
        element = self.click(locator, wait, timeout, error_message)
        element.send_keys(string)
        return element

