from config import WEIGHTS_PATH, INPUT_SHAPE, NUM_CLASSES, ACTIVE_MODEL
from models.custom_xception import build_custom_xception

model = None

def load_model_from_config():
    global model
    if ACTIVE_MODEL == "custom_xception":
        model = build_custom_xception(INPUT_SHAPE, NUM_CLASSES)
        model.load_weights(WEIGHTS_PATH)
    else:
        raise ValueError(f"Unknown model: {ACTIVE_MODEL}")

load_model_from_config()
