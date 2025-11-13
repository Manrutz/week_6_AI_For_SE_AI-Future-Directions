
# Raspberry Pi inference loop example using tflite-runtime (or tensorflow)
import time, cv2, numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

interpreter = Interpreter('model_recyclables_quant.tflite')
interpreter.allocate_tensors()
input_idx = interpreter.get_input_details()[0]['index']
output_idx = interpreter.get_output_details()[0]['index']

cam = cv2.VideoCapture(0)
IMG_SIZE = (160,160)
while True:
    ret, frame = cam.read()
    if not ret:
        break
    img = cv2.resize(frame, IMG_SIZE)
    img_uint8 = img.astype(np.uint8)[np.newaxis, ...]
    start = time.time()
    interpreter.set_tensor(input_idx, img_uint8)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_idx)
    latency = (time.time() - start) * 1000
    label = int(np.argmax(preds[0]))
    print(f"Pred: {label} | Latency: {latency:.1f} ms")
