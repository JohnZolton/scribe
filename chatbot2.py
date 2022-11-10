"""
v2 of chatbot. Figured I don't need the mic always on since its a dialogue
"""
import sounddevice as sd
import wavio as wv
import whisper
import timeit
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


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
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")


# Let's chat for 5 lines
for step in range(5):
    message = record()
    message = transcribe(whisper_model)
    print('Heard: ', message)
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
