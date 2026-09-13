import joblib

from src.features import apply_combo_map

MODEL_PATH = "models/model.pkl"

_model = None

def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model

def predict(feature_df):
    model = load_model()
    pipeline = model['pipeline']
    combo_map = model['combo_map']

    encoded = apply_combo_map(feature_df, combo_map)
    probabilities = pipeline.predict_proba(encoded)[:, 1]
    return probabilities