
# Task 1 — Edge TinyML Recyclables Classifier — Report

**Student:** Remmy Kipruto Tumo

## Dataset
Use TrashNet or similar dataset with classes: plastic, glass, metal, paper, non-recyclable.

## Training
- Model: MobileNetV2 backbone, small classification head
- Image size: 160x160
- Training: transfer learning (base frozen) -> fine-tune last layers
- Epochs: 10-15 (example)

## Results (example placeholders)
- Validation accuracy: *to be filled after training* (expected 80-95% depending on data)
- Model size (.tflite): ~0.5 - 4 MB (after quantization)
- Inference latency (Raspberry Pi 4): ~30-200 ms depending on model size and interpreter

## Deployment steps
1. Train in Colab and save `model_recyclables.h5`.
2. Convert to TFLite using `src/convert_tflite.py` or notebook cell.
3. Transfer `model_recyclables_quant.tflite` to Raspberry Pi: `scp model_recyclables_quant.tflite pi@<ip>:/home/pi/`
4. Install runtime: `pip install tflite-runtime` (or `pip install tensorflow` on capable Pi).
5. Run `python3 src/inference_pi.py` and inspect outputs. Use camera or image folder for testing.

## Limitations & Future Work
- Collect more labeled images to improve generalization.
- Use on-device calibration and continual learning for new waste types.
- Optimize model further via pruning and knowledge distillation for microcontrollers.
