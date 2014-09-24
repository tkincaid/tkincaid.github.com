from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from page import Locator
from datarobot_page import DataRobotPage

class DataPage(DataRobotPage):

    LOADED_ELEMENT = Locator(By.ID, 'eda')
    SUMMARY_TAB = Locator(By.CSS_SELECTOR, '#EDA_menu > li:nth-child(1)')
    SINGLE_VAR_TAB = Locator(By.CSS_SELECTOR, '#EDA_menu > li:nth-child(2)')
    RAW_DATA_TAB = Locator(By.CSS_SELECTOR, '#EDA_menu > li:nth-child(3)')
    VAR_TYPES_ELEMENTS = Locator(By.CSS_SELECTOR, '.vartype')

    SINGLE_VAR_SORT_BUTTON = Locator(By.CSS_SELECTOR, '#plot > label > input[type="checkbox"]')
    #SUMMARY_COLUMNS = Locator()

    def __init__(self, selenium):
        super(DataPage, self).__init__(selenium)

    def get_varTypes(self):
        """Returns a list of all the var types on the EDA table"""

        varTypeElements = self.selenium.find_elements_by_css_selector(self.VAR_TYPES_ELEMENTS.value)
        return map(lambda x: x.text, varTypeElements)

    def go_to_summary(self):
        self.click(self.SUMMARY_TAB)

    def go_to_single_var(self):
        self.click(self.SINGLE_VAR_TAB)

    def go_to_raw_data(self):
        self.click(self.RAW_DATA_TAB)

    def click_column_name(self, name):
        # Used to search for and click a specific column name
        raise NotImplementedError

    def click_column_index(self, index):
        """
        Clicks a dataset in the EDA table by its index

        Note: index must be >= 1
        """

        # TODO: Throw an error message if index is out of range
        location = Locator(By.CSS_SELECTOR, '#eda_data tr:nth-child(%d)>.column-name span' % (index))
        return self.click(location)

    def single_var_go_to_next(self, column):
        column.send_keys(Keys.ARROW_DOWN)

    def single_var_sort(self, state):
        """
        Enables or disables sorting the single variable chart

        Note: the sort button only shows up under certain conditions
        """
        pass
        #self.click(self.SINGLE_VAR_SORT_BUTTON)
