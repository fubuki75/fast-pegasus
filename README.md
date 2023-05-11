# fast-Pegasus

### Reduction of Pegasus model size by 3X, and boost in inference speed up to 2-3X
  Pegasus implementation of the fastT5 library (https://github.com/Ki6an/fastT5) and fastBart (https://github.com/siddharth-sharma7/fast-Bart). 
  fast-pegasus is very similar to fast-Bart, thanks to the object-oriented design of transformers models.
  
  **Pytorch model -> ONNX model -> Quantized ONNX model**

---
## Install

Install using requirements.txt file.

```bash
git clone https://github.com/fubuki75/fast-pegasus
cd fast-Pegasus
pip install -r requirements.txt
```
---
## Usage

The `export_and_get_onnx_model()` method exports the given pretrained Pegasus model to onnx, quantizes it and runs it on the onnxruntime with default settings. The returned model from this method supports the `generate()` method of huggingface.

> If you don't wish to quantize the model then use `quantized=False` in the method.

```python
from fastPegasus import OnnxPegasus, export_and_get_onnx_model, get_onnx_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_or_model_path = "google/pegasus-cnn_dailymail"

model = export_and_get_onnx_model(model_or_model_path)

tokenizer = AutoTokenizer.from_pretrained(model_or_model_path)

t_input = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest..."

token = tokenizer(t_input, return_tensors="pt")

input_ids = token["input_ids"]
attention_mask = token["attention_mask"]

# 'set num_beams = 1' for greedy search
tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=2)
output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
print(output)
```

> to run the already exported model use `get_onnx_model(model_name, onnx_models_path)`

you can customize the whole pipeline as shown in the below code example:

```python
from fastPegasus import OnnxPegasus,export_and_get_onnx_model,get_onnx_model,get_onnx_runtime_sessions,generate_onnx_representation,quantize
from transformers import AutoTokenizer

model_or_model_path = "google/pegasus-cnn_dailymail"

# Step 1. convert huggingfaces pegasus model to onnx
onnx_model_paths = generate_onnx_representation(model_or_model_path)

# Step 2. (recommended) quantize the converted model for fast inference and to reduce model size.
quant_model_paths = quantize(onnx_model_paths)

# step 3. setup onnx runtime,
model_sessions = get_onnx_runtime_sessions(quant_model_paths)

# step 4. get the onnx model，
model = OnnxPegasus(model_or_model_path, model_sessions)

                      ...
```
##### custom output paths 
By default, fastBart creates a `model` folder in the current directory and stores all the models. You can provide a custom path for a folder to store the exported models. And to run already `exported models` that are stored in a custom folder path: use `get_onnx_model(onnx_models_path="/path/to/custom/folder/")`

```python
from fastPegasus import export_and_get_onnx_model, get_onnx_model

model_name = "google/pegasus-cnn_dailymail"
custom_output_path = "/path/to/custom/folder/"

# 1. store models to custom_output_path
model = export_and_get_onnx_model(model_name, custom_output_path)

# 2. run already exported models that are stored in custom path
# model = get_onnx_model(model_name, custom_output_path)
```
## Some details of `past_key_values`
In addition to the implementation details mentioned in fastT5, here we discuss about `past_key_values`
* Why do the transformers models maintain `past_key_values`? 

`past_key_values` is a common technique adopted in transformers models. like T5, Bart and Pegasus. According to the [official documentations](https://huggingface.co/docs/transformers/main/en/model_doc/t5#transformers.T5ForConditionalGeneration.forward.past_key_values):

`past_key_values` `(tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

* How does this param speed up decoding process?

`past_key_values` stores the previous projected keys and values, to reduce the computation cost of multi-head projection.
 
original multi-head projection: project `(batch,dec_len,hs)` to `(batch,n_heads,dec_len,d_v)`

multi-head projection with `past_key_values`: 
1. first project `(batch,1,hs)` to `(batch,n_heads,1,d_v)`
2. concat it and `past_key_values` `(batch,n_heads,dec_len-1,d_v)`

similar to the toy example code below: 
```python
import numpy as np
"""
k,v: represents input key and values, shape is [1, d_model].
past_key_value: [past_k,past_v] [2,dec_len-1,d_v]
"""
# first project
k = np.matmul(k, W_k)
v = np.matmul(v, W_v)

if past_key_value is not None: # then concat
    k = np.concatenate([past_key_value[0], k], axis=0)
    v = np.concatenate([past_key_value[1], v], axis=0)
```
* Why `past_key_values` will not be influenced by new tokens?

Because the attention of decoder block is actually an upper-triangle-shaped mask, which prevent the previous tokens attending to new ones.

## Inference speed test results
I test the inference speed locally, more details in `runtime_test.py`. My CPU is AMD Ryzen 7 5800X 8-Core Processor 3.80 GHz.

| param & results           | 1 | 2 | 3     | 4 | 5 |
|---------------------------| --- | --- |-------| --- | --- |
| batch_size                | 1 | 1 | 2     | 4 | 4 |
| beam_size                 | 1 | 4| 4     |1|4|
| onnx_quantized_time(s)    |1.72|4.05| 6.02  |4.61|13.57|
| onnx_time           (s)   |4.06|7.74| 11.84 |6.68|21.86|
| pytorch_time          (s) |4.48|11.00| 11.34 |10.50|18.56|

I found that onnx model running on onnxruntime perform better than pytorch model, when the amount of parallel computation
is relatively low, like `batch_size*beam_size<8`. However, I am not sure why this happens, so please pay more attention on
param settings when using onnx models without quantization.

## Functionalities

- Export any pretrained Pegasus model to ONNX easily.
- The exported model supports beam search and greedy search and more via `generate()` method.
- Reduce the model size by `3X` using quantization.
- Up to `2-3X` speedup compared to PyTorch execution for greedy search and `2-3X` for beam search.

