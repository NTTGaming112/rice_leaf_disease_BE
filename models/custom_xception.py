import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# --- Custom Xception ---
def separable_conv_block(x, filters, kernel_size=3, strides=2):
    residual = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(x)
    main_path = layers.ReLU()(x)
    main_path = layers.SeparableConv2D(filters, (kernel_size, kernel_size), padding='same')(main_path)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.ReLU()(main_path)
    main_path = layers.SeparableConv2D(filters, (kernel_size, kernel_size), padding='same')(main_path)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.MaxPooling2D((3, 3), strides=strides, padding='same')(main_path)
    output = layers.Add()([residual, main_path])
    return output

def build_custom_xception(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2)
    ], name='data_augmentation')
    x = augmentation(inputs)
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', name='entry_conv')(x)
    x = layers.BatchNormalization(name='entry_bn')(x)
    x = layers.ReLU(name='entry_relu')(x)
    x = separable_conv_block(x, filters=128, kernel_size=3, strides=2)
    x = separable_conv_block(x, filters=256, kernel_size=3, strides=2)
    x = separable_conv_block(x, filters=512, kernel_size=3, strides=2)
    x = layers.SeparableConv2D(1024, (3, 3), padding='same', name='exit_sepconv')(x)
    x = layers.BatchNormalization(name='exit_bn')(x)
    x = layers.ReLU(name='exit_relu')(x)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output_layer')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name='Custom_Xception_1.2M')
    return model
