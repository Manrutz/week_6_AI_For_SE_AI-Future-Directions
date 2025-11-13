
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def build_model(img_size=(160,160,3), num_classes=5):
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_size, include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.Input(shape=img_size)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print('This script is a helper for training. Use the notebook for Colab-run training.')
