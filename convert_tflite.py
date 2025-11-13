
import tensorflow as tf
import numpy as np

def convert_keras_to_tflite(model_path='model_recyclables.h5', output_path='model_recyclables_quant.tflite', train_ds=None, img_size=(160,160)):
    model = tf.keras.models.load_model(model_path)
    def representative_data_gen():
        if train_ds is None:
            for _ in range(100):
                data = np.random.rand(1, img_size[0], img_size[1], 3).astype(np.float32)
                yield [data]
        else:
            for images, labels in train_ds.take(100):
                batch = tf.image.resize(images, img_size)
                batch = tf.cast(batch, tf.float32)
                yield [batch.numpy()]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print('Saved', output_path)

if __name__ == '__main__':
    convert_keras_to_tflite()
