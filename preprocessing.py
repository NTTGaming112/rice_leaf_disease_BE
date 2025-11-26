import numpy as np
from scipy.signal import wiener
import cv2
from skimage.filters import threshold_otsu
from config import PREPROCESSING_PIPELINES, ACTIVE_MODEL

def apply_wiener_filter(image, mysize=5):
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        filtered_image[:, :, i] = wiener(image[:, :, i], mysize=mysize)
    return filtered_image

def apply_otsu_segmentation(image):
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    thresh_value = threshold_otsu(gray)
    binary_mask = gray > thresh_value
    blurred_background = cv2.GaussianBlur(image, (21, 21), 0)
    segmented_image = image.copy()
    binary_mask_3d = np.expand_dims(binary_mask, axis=-1)
    segmented_image = np.where(binary_mask_3d, segmented_image, blurred_background)
    return segmented_image

def preprocess_image(image, wiener_size=5):
    """
    Pipeline động: chọn các bước theo PREPROCESSING_PIPELINES và ACTIVE_MODEL trong config
    """
    processed_image = image.copy().astype(np.float32)
    pipeline = PREPROCESSING_PIPELINES.get(ACTIVE_MODEL, [])
    if "wiener" in pipeline:
        processed_image = apply_wiener_filter(processed_image, mysize=wiener_size)
        processed_image = np.clip(processed_image, 0, 255)
    if "otsu" in pipeline:
        processed_image = apply_otsu_segmentation(processed_image.astype(np.uint8))
    # Nếu không có bước nào thì chỉ chuẩn hóa
    return processed_image.astype(np.float32) / 255.0
