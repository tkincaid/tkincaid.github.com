import time
from functools import wraps

from tests.pages.aim_page import AimPage
from tests.pages.data_page import DataPage
from tests.pages.join_page import JoinPage
from tests.pages.leaderboard_page import LeaderboardPage
from tests.pages.queue_page import QueuePage
from tests.ui.ui_test_base import UITestBase

# TODO: Move this to a more general location for utilities
def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck, e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print msg
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry

class TestAppUI(UITestBase):

    def test_new_user(self):
        # First register for an account
        join_page = JoinPage(self.browser)
        join_page.go_to_join_page(self.base_url, self.username, self.invite_code)
        join_page.sign_up(self.password)
        join_page.is_in_upload_page()

        # Upload a dataset
        self.frontpage.upload_url('https://s3.amazonaws.com/datarobot_test/kickcars-sample-200.csv')

        # Check that the EDA page loaded properly
        self.datapage = DataPage(self.browser)
        self.datapage.verify_is_loaded(error_message = "EDA is not visible")

        self.assert_varTypes_exist()

        # View some of the EDA charts
        column = self.datapage.click_column_index(1)

        # XXX: Eliminate sleep by checking for chart
        time.sleep(2)

        self.datapage.single_var_go_to_next(column)
        self.datapage.single_var_go_to_next(column)
        self.datapage.single_var_go_to_next(column)

        # Kickoff some models
        aimpage = AimPage(self.browser)
        aimpage.select_target_variable('IsBadBuy')
        aimpage.select_metric('Ranking')
        aimpage.set_modeling_mode(AimPage.SEMI_AUTO_BUTTON)
        aimpage.click_start_button()

        # View the leaderboard
        self.frontpage.select_models_tab()

        leaderboard_page = LeaderboardPage(self.browser)
        leaderboard_page.verify_is_loaded(error_message = "Leaderboard is not visible")
        # ACTION:   Click on the Modeling menu item after at least one item is complete
        # EXPECTED: The leaderboard displays
        model_data_id = leaderboard_page.select_model(index = 0, timeout = 30)
        model_title = leaderboard_page.get_model_title(model_data_id)
        # ACTION:   Click on the plus icon under the Training Sample Size column of the first item on the board
        # EXPECTED: A small popup appears showing a percent value

        # ACTION:   Change to x% and click Run
        # EXPECTED: The popup closes
        # EXPECTED: A new item is added to the queue
        new_sample_size = 80
        leaderboard_page.submit_sample_size(model_data_id = model_data_id, sample_size = new_sample_size)

        queue = QueuePage(self.browser, timeout = 5)
        worker_count = queue.get_worker_count()
        self.assertGreater(worker_count, 0)

        item = queue.get_item(model_title, new_sample_size)
        self.assertIsNotNone(item)

        # Check that active and processing are not empty
        self.assertTrue(len(queue.get_processing_models()) != 0)
        self.assertTrue(len(queue.get_waiting_models()) != 0)

        # Check to see if the resources are updating (big timeout, with small datasets some go so quick the screen doesn't update)
        queue.check_resource_usage()

        while(worker_count > 0):
            queue.decrease_worker()
            new_worker_count = queue.get_worker_count()
            self.assertLess(new_worker_count, worker_count)
            worker_count = new_worker_count

        # ACTION:   Click on Single Model
        # EXPECTED: Lift Chart is selected
        # EXPECTED: A graph is displayed
        # ACTION:   Click ROC curve
        # EXPECTED: A graph is displayed
        # ACTION:   Click Model Info
        # EXPECTED: The following information is displayed: Blueprint, Tuning Parameters, Sample Size, Run Time

    @retry(AssertionError, tries=15, delay=1, backoff=1)
    def assert_varTypes_exist(self):
        # Check that all the var types are not empty
        # Github Issue 1302
        varTypes = self.datapage.get_varTypes()

        # Filter out empty strings
        varTypes = filter(None, varTypes)

        # Assert that not all the var types are empty
        self.assertNotEqual(len(varTypes), 0, "Var types could not be found")
