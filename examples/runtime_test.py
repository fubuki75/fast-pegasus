from fastPegasus import (
    OnnxPegasus,
    export_and_get_onnx_model,
    get_onnx_model,
    get_onnx_runtime_sessions,
    generate_onnx_representation,
    quantize,
)

from transformers import PegasusTokenizer, AutoModelForSeq2SeqLM
from timeit import default_timer as timer

model_or_model_path = 'google/pegasus-cnn_dailymail'

#onnx quantized
model_quantized = export_and_get_onnx_model(model_or_model_path, quantized=True)

#onnx
model_onnx = export_and_get_onnx_model(model_or_model_path, quantized=False)

#pytorch model
py_model = AutoModelForSeq2SeqLM.from_pretrained(model_or_model_path)

# python tokenizer
tokenizer = PegasusTokenizer.from_pretrained(model_or_model_path)

# test params
batch_size = 1                       # test batch size
num_beams = 1                         # Number of beams per input text
max_encoder_length = 256               # Maximum input token length
max_decoder_length = 128               # Maximum output token length

def infer(model, tokenizer, text):
    # Truncate and pad the max length to ensure that the token size is compatible with fixed-sized encoder (Not necessary for pure CPU execution)
    batch = tokenizer(text, max_length=max_decoder_length, truncation=True, padding='max_length', return_tensors="pt")
    output = model.generate(**batch, max_length=max_decoder_length, num_beams=num_beams, num_return_sequences=num_beams,
                            temperature=1.0, repetition_penalty=1.5)
    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]

    print('Texts:')
    for i, summary in enumerate(results):
        print(i + 1, summary)

sentences = [
    '''(CNN)Deion Sanders is such a dad. The NFL legend called out Deion Sanders Jr. on Twitter for saying he only eats "hood doughnuts." In response, the elder Sanders -- in front of his 912,000 followers -- reminded his son he has a trust fund, a condo and his own clothing line called "Well Off." "You're a Huxtable with a million $ trust fund. Stop the hood stuff!" Sanders followed it up with another tweet that included the hashtags #versacesheets #Huxtable and #Trustfund. Junior is a wide receiver at Southern Methodist University, an aspiring entrepreneur and occasional rapper. His Twitter timeline is a mix of biblical verses, motivational quotes and references to sports, cars, school and Balenciaga shoes. He also has gone on record with his love for "hood doughnuts," or confections from "a place in the hood," saying "if my doughnuts don't come in a plain white box, I don't want them!" His father promptly put him in his place. Sanders Jr. seemed to take the public browbeating in stride, retweeting his father's comments. At least he knew better than to delete them.''',
    '''(CNN)"Jake the dog and Finn the human. The fun will never end. Adventure Time." So begins the dreamy theme song intro to the strangely addictive Cartoon Network TV show that's centered around psychedelic characters like the Ice King, Marceline the Vampire Queen and, of course, Jake and Finn. Now, mega-fans of the hit show can experience "Adventure Time" in the skies. Thai Smile, a subsidiary of Thailand flag carrier Thai Airways, on Thursday unveiled colorful new livery featuring Jake, Finn and the beloved Princess Bubblegum sprawled across an Airbus A320 at Bangkok's Suvarnabhumi International Airport. The interior of the plane also has an Adventure Time theme, with overhead bins, head rests and even air sickness bags covered in the faces of characters from the show. Airlines show off their new flying colors . The Adventure Time plane is the result of a partnership between Thai Airways subsidiary Thai Smile and Cartoon Network Amazone, a new water park near the Thai resort city of Pattaya featuring attractions based on shows that appear on the Turner Broadcasting System channel. Turner Broadcasting is a parent company of CNN. Check out these cool airline liveries . The inaugural Thai Smile Adventure Time flight takes place on April 4, heading from Bangkok to Phuket.''',
    '''(CNN)Five Americans who were monitored for three weeks at an Omaha, Nebraska, hospital after being exposed to Ebola in West Africa have been released, a Nebraska Medicine spokesman said in an email Wednesday. One of the five had a heart-related issue on Saturday and has been discharged but hasn't left the area, Taylor Wilson wrote. The others have already gone home. They were exposed to Ebola in Sierra Leone in March, but none developed the deadly virus. They are clinicians for Partners in Health, a Boston-based aid group. They all had contact with a colleague who was diagnosed with the disease and is being treated at the National Institutes of Health in Bethesda, Maryland. As of Monday, that health care worker is in fair condition. The Centers for Disease Control and Prevention in Atlanta has said the last of 17 patients who were being monitored are expected to be released by Thursday. More than 10,000 people have died in a West African epidemic of Ebola that dates to December 2013, according to the World Health Organization. Almost all the deaths have been in Guinea, Liberia and Sierra Leone. Ebola is spread by direct contact with the bodily fluids of an infected person.''',
    '''London (CNN)A 19-year-old man was charged Wednesday with terror offenses after he was arrested as he returned to Britain from Turkey, London's Metropolitan Police said. Yahya Rashid, a UK national from northwest London, was detained at Luton airport on Tuesday after he arrived on a flight from Istanbul, police said. He's been charged with engaging in conduct in preparation of acts of terrorism, and with engaging in conduct with the intention of assisting others to commit acts of terrorism. Both charges relate to the period between November 1 and March 31. Rashid is due to appear in Westminster Magistrates' Court on Wednesday, police said. CNN's Lindsay Isaac contributed to this report.'''
]

onnx_quantized_time = []
onnx_time = []
pytorch_time = []

sentence_batchs = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

for i in range(5):
    for s in sentence_batchs:

        start = timer()
        infer(model_quantized, tokenizer, s)
        end = timer()
        onnx_quantized_time.append(end-start)

        start = timer()
        infer(model_onnx, tokenizer, s)
        end = timer()
        onnx_time.append(end-start)

        start = timer()
        infer(py_model, tokenizer, s)
        end = timer()
        pytorch_time.append(end-start)


print("Onnx quantized time", sum(onnx_quantized_time)/len(onnx_quantized_time))
print("Onnx time:", sum(onnx_time)/len(onnx_time))
print("PyTorch time:", sum(pytorch_time)/len(pytorch_time))

