# Cấu hình cho backend
CLASS_NAMES = ['N', 'P', 'K']
INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = len(CLASS_NAMES)

WEIGHTS_PATH = "models/model_fold_5.weights.h5"

PREPROCESSING_PIPELINES = {
    "custom_xception": ["none"]
}

# Model đang sử dụng
ACTIVE_MODEL = "custom_xception"
