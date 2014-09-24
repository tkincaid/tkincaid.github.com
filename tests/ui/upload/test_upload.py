from tests.ui.ui_test_base import UITestBase

class UITestUpload(UITestBase):

    def test_upload_kickcars(self):
        self.frontpage.login(self.base_url, self.username, self.password)
        self.frontpage.upload_url('https://s3.amazonaws.com/datarobot_test/kickcars-sample-200.csv')

    # TODO: Try uploading different datasets
