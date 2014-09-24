import pytest
from flask import json


@pytest.mark.db
def test_total_models(app_test_client):
    response = app_test_client.get('/public/total-models')
    assert response.status_code == 200
    assert response.mimetype == 'application/json'

    data = json.loads(response.data)

    keys = data.keys()
    # Sorting since the order of the keys is not guaranteed.
    keys.sort()
    assert keys == ['total-models']
    assert isinstance(data['total-models'], int)
