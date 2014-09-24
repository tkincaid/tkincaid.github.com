import config.test_config
from config.engine import EngConfig
# Enable the progress API for the stand-alone test API
EngConfig['TEST_MODE'] = False
from MMApp.api import app, root, queue, create_leaderboard_item, save_leaderboard_item, report_complete, report_error, get_data_url

