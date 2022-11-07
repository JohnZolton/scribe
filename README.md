This is a psuedo-live transcription program using Whisper. 

A microphone recorder function and a whisper function run in parallel using multiprocessing.
The microphone continuously records 5 second audio files while the whisper function checks if a new audio file is ready for transcription and then converts speech to text.


I then used this pseudo-live transcription for a quesiton-answering bot. The inspiration was a companion bot for students, it detects questions (possibly from a teacher/professor) and finds the answer from an accompanying text file (like a notebook). In testing it correctly answered simple fact questions (like the ruling in a court case) but failed to answer more abstract questions (like "what was the defense's argument?").  
