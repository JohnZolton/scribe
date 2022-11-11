"""
v2 of chatbot. Figured I don't need the mic always on since its a dialogue
"""
import sounddevice as sd
import wavio as wv
import whisper
import timeit
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration



def record():
    # Sampling frequency
    freq = 44100
    # Recording duration
    duration = 5
    print('**listening**')
    
    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    sd.wait()
    # Convert the NumPy array to audio file
    wv.write("recording0.wav", recording, freq, sampwidth=2)

def transcribe(model):
    result = model.transcribe("recording0.wav", fp16=False)
    return result['text']

whisper_model = whisper.load_model("small.en")

#download and setup the model and tokenizer
model_name = 'facebook/blenderbot-400M-distill'
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)


for i in range(5):
    print()
while True:
    message = record()
    message = transcribe(whisper_model)
    print('USER: ', message)

    #tokenize the utterance
    inputs = tokenizer(message, return_tensors="pt")
    result = model.generate(**inputs, max_new_tokens=1000)
    response = tokenizer.decode(result[0])
    print('CHATBOT: ', response[3:-4])
