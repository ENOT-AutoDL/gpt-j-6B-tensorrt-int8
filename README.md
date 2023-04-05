# GPT-J 6B inference on TensorRT with INT-8 precision

Repository contains inference example and accuracy validation of quantized GPT-J 6B TensorRT model.
Prebuilt TensorRT engines are published on [Hugging Face](https://huggingface.co/ENOT-AutoDL/gpt-j-6B-tensorrt-int8)
:hugs:.
Our example notebook automatically downloads the appropriate engine.

**Currently published engines for the following GPUs only:**
* RTX 2080 Ti
* RTX 3080 Ti
* RTX 4090

**ONNX model and build script will be published later.**

## Metrics:

|   |TensorRT INT8+FP32|torch FP16|torch FP32|
|---|:---:|:---:|:---:|
| **Lambada Acc** |78.79%|79.17%|-|
| **Model size (GB)**  |8.5|12.1|24.2|

### Test environment

* GPU RTX 4090
* CPU 11th Gen Intel(R) Core(TM) i7-11700K
* TensorRT 8.5.3.1
* pytorch 1.13.1+cu116

## Latency:

|Input sequance length|Number of generated tokens|TensorRT INT8+FP32 ms|torch FP16 ms|Acceleration|
|:---:|:---:|:---:|:---:|:---:|
|64|64|1040|1610|1.55|
|64|128|2089|3224|1.54|
|64|256|4236|6479|1.53|
|128|64|1060|1619|1.53|
|128|128|2120|3241|1.53|
|128|256|4296|6510|1.52|
|256|64|1109|1640|1.49|
|256|128|2204|3276|1.49|
|256|256|4443|6571|1.49|

### Test environment

* GPU RTX 4090
* CPU 11th Gen Intel(R) Core(TM) i7-11700K
* TensorRT 8.5.3.1
* pytorch 1.13.1+cu116
