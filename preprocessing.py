import numpy as np

def preprocess_image(image, wiener_size=5):
    """
    Tiền xử lý cho custom_xception: chỉ chuẩn hóa về [0, 1]
    """
    processed_image = image.copy().astype(np.float32)
    return processed_image / 255.0
