from fastPegasus import OnnxPegasus, export_and_get_onnx_model, get_onnx_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_or_model_path = "google/pegasus-cnn_dailymail"

model = export_and_get_onnx_model(model_or_model_path)

# if you've already exported the models
#model = get_onnx_model(model_name=model_or_model_path, onnx_models_path='./models')

tokenizer = AutoTokenizer.from_pretrained(model_or_model_path)

t_input = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest‘" \
          " structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction," \
          " the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world," \
          " a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the " \
          "first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of " \
          "the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, " \
          "the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."

token = tokenizer(t_input, return_tensors="pt")

input_ids = token["input_ids"]
attention_mask = token["attention_mask"]

# 'set num_beams = 1' for greedy search
tokens = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=2)

output = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

print(output)
