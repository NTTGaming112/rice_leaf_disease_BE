from PIL import Image
import numpy as np
import io
from config import INPUT_SHAPE
from preprocessing import preprocess_image

def preprocess_image_for_model(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img_array = np.array(image)
    img_array = preprocess_image(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
