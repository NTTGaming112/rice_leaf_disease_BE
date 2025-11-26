# Cấu hình cho backend

CLASS_NAMES = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast', 'Leaf scald', 'Sheath Blight']
INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = len(CLASS_NAMES)

# Đường dẫn model (có thể thay đổi cho từng model)
KERAS_MODEL_PATH = "models/basic_capsnet_best.keras"
WEIGHTS_PATH = "models/model_fold_5.weights.h5"

# Định nghĩa pipeline preprocessing cho từng model
PREPROCESSING_PIPELINES = {
    "capsnet": ["wiener", "otsu"],
    "custom_xception": ["none"]
}

# Model đang sử dụng
ACTIVE_MODEL = "capsnet"
