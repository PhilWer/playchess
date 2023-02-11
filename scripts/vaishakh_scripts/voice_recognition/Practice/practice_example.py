'''
packages too install 
1.pip3 install SpeechRecognition
2.sudo apt-get install python-pyaudio python3-pyaudio # for using Micro phones
    2.1. pip install pyaudio
'''

import speech_recognition as sr

print(sr.__version__)

# create recodnizer instance
r = sr.Recognizer()

# capturing data usinf record
harvard = sr.AudioFile(
    '/home/vaishakh/Back_up/playchess/scripts/vaishakh_scripts/Voice_recognition/Practice/python-speech-recognition/audio_files/intro-music-black-box-roughly-made-bass-house-13217.wav')
with harvard as source:
    # SciPY ti preprocess audio data
    # deal with ambient noise but consumes by default 1s of source
    r.adjust_for_ambient_noise(source)
    audio1 = r.record(source, duration=4)  # dutation default full
    # resumes audio1 when inted in same with
    audio2 = r.record(source, offset=4, duration=4)
    print(f'type of audio{type(audio1)}')

# invoke recognizer to recognize speech in audio
# how all to give full respone than most likely transcription
r.recognize_google(audio1, show_all=True)
