import os
def test_model_exists():
    assert os.path.exists("models/rf_model.pkl")