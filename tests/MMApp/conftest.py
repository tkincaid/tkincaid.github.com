import pytest
from mock import patch

from config.engine import EngConfig


@pytest.fixture(scope="session")
def app_test_client():
    """A Flask test_client instance for app.py"""

    with patch.dict(EngConfig, {'TEST_MODE': True}, clear=False):
        import MMApp.app
        return MMApp.app.app.test_client()
