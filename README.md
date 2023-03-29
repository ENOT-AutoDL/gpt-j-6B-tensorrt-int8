# Demo of usage gpt-j-6B-trt-int8 model

This repository contains example of inference and accuracy test of quantized gpt-j-6b tensorrt model. Prebuilded engines published [here](https://huggingface.co/ENOT-AutoDL/gpt-j-6B-tensorrt-int8). The demo script downloads the appropriate engine automatically.

**Currently published engines only for next GPUs:**
* rtx2080ti
* rtx3080ti
* rtx4090

**onnx + build script will be published later.**

## Test result

|   |INT8|FP32|
|---|:---:|:---:|
| **Lambada Acc** |78.50%|79.54%|
| **Model size (GB)**  |8.1|23|
