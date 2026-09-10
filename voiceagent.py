import ollama
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import speech_recognition as sr
import pyttsx3
import io

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 175) 

def speak(text):
    print(f"\nAI: {text}")
    engine.say(text)
    engine.runAndWait()

def listen_to_user():
    # Recording configuration parameters
    sample_rate = 16000  
    duration = 5  # Listens in 5-second blocks
    
    print("\nListening... Speak now.")
    # Record audio data block into system RAM
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording block is filled
    
    # Pack the raw recording matrix into a standard WAV format stream
    wav_io = io.BytesIO()
    wavfile.write(wav_io, sample_rate, recording)
    wav_io.seek(0)
    
    # Feed audio data into Speech Recognizer
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except Exception:
            return None

speak("System operational. Ready for input.")
while True:
    user_input = listen_to_user()
    if user_input:
        if user_input.lower() in ['stop', 'exit', 'bye']:
            speak("Goodbye!")
            break
            
        response = ollama.chat(
            model='LiquidAI/lfm2.5-1.2b-instruct',
            messages=[{'role': 'user', 'content': user_input}]
        )
        speak(response['message']['content'])