"""
Program transcribes text using Whisper
Once question is detected it uses a transformer model to answer it
"""

# import required libraries
import sounddevice as sd
import wavio as wv
import whisper
import multiprocessing
from transformers import pipeline
import os


def record(conn):
    # Sampling frequency
    freq = 44100
    # Recording duration
    duration = 5
    print('Recording')
    
    while True:
        newmsg = False
        conn.send(newmsg)
        # Start recorder with the given values
        # of duration and sample frequency
        recording = sd.rec(int(duration * freq),
                        samplerate=freq, channels=2)

        # Record audio for the given number of seconds
        sd.wait()
        # i wonder if i can bounce between two files so i don't lose words

        # Convert the NumPy array to audio file
        wv.write("recording0.wav", recording, freq, sampwidth=2)
        newmsg = True
        conn.send(newmsg)

def transcribe(conn, to_question, model):

    while True:
        while True:
            newmsg = conn.recv()
            if newmsg:
                break
        
        audio = whisper.load_audio("recording0.wav")
        audio = whisper.pad_or_trim(audio)
        
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(language= 'en', fp16=False)
        result = whisper.decode(model, mel, options)
        if result.text[-1] == '?':
            to_question.send(result.text)

def loadfile(directory):
    content = {}

    for file in os.scandir(directory):
        with open(file) as f:
            filename = os.path.split(file)[1]
            contents = f.read()
            content[filename] = contents
    return content

def answerquestion(conn, qa_model, context):
    while True:
        while True:
            question = conn.recv()
            if question: break
        # TODO pause the other processes while finding answer, might speed up?
        ans = qa_model(question = question, context = context)
        print(question)
        print('ans: ', ans['answer'])


if __name__=="__main__":
    model = whisper.load_model("small.en")

    # set question answering model, loading context
    qa_model = pipeline("question-answering")
    file = 'Corpus'
    doc = loadfile(file)
    for page in doc:
        context = doc[page]


    to_mic, to_whisper = multiprocessing.Pipe()
    from_whisper, to_question = multiprocessing.Pipe()

    mic = multiprocessing.Process(target=record, args = (to_whisper,))
    whisp = multiprocessing.Process(target=transcribe, args = (to_mic, to_question, model))
    QA = multiprocessing.Process(target=answerquestion, args=(from_whisper, qa_model, context))
    
    mic.start()
    whisp.start()
    QA.start()

    mic.join()
    whisp.join()
    QA.join()