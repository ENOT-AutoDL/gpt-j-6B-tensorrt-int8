# GPT-J 6B model inference on TensorRT with INT-8 precision.

Repository contains inference example and accuracy validation of quantized GPT-J 6B TensorRT model. Prebuilt TensorRT engines are published on [Hugging Face](https://huggingface.co/ENOT-AutoDL/gpt-j-6B-tensorrt-int8):hugs:. The example script automatically downloads the appropriate engine.

**Currently published engines only for next GPUs:**
* RTX 2080 Ti
* RTX 3080 Ti
* RTX 4090

**ONNX model and build script will be published later.**

## Validation table

|   |INT8|FP32|
|---|:---:|:---:|
| **Lambada Acc** |78.50%|79.54%|
| **Model size (GB)**  |8|23|
