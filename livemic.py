"""
Okay so the plan is to record from the mic
into a continuous, rolling 30 second window
and feed that into whisper
or just like 5 second chunks since processing time is under a second
"""

# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import whisper
import timeit

model = whisper.load_model("base")

# Sampling frequency
freq = 44100

# Recording duration
duration = 5
print('Recording')
text = ""
while True:
# Start recorder with the given values
# of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                    samplerate=freq, channels=2)

    # Record audio for the given number of seconds
    sd.wait()
    # i wonder if i can bounce between two files so i don't lose words

    # Convert the NumPy array to audio file
    wv.write("recording0.wav", recording, freq, sampwidth=2)


    audio = whisper.load_audio("recording0.wav")
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language= 'en', fp16=False)
    result = whisper.decode(model, mel, options)

    text += result.text
    print(result.text)
    print()
# i need to run the recording and the transcribing in parallel so i don't lose words