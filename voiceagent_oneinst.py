import os
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppresses sounddevice engine warnings

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
    sample_rate = 16000  
    duration = 5  # Listens for a 5-second block
    
    print("\nListening... Speak your instruction now.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is done
    
    wav_io = io.BytesIO()
    wavfile.write(wav_io, sample_rate, recording)
    wav_io.seek(0)
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except Exception:
            print("Could not understand the audio.")
            return None

# --- SINGLE EXECUTION FLOW ---

# 1. Listen for exactly one instruction
user_input = listen_to_user()

if user_input:
    try:
        # 2. Get response from local Liquid AI LFM model
        response = ollama.chat(
            model='LiquidAI/lfm2.5-1.2b-instruct',
            messages=[{'role': 'user', 'content': user_input}]
        )
        
        # 3. Speak the answer out loud
        ai_reply = response['message']['content']
        speak(ai_reply)
        
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
else:
    speak("No clear instruction was detected. Exiting.")

# The script now automatically ends here and closes back to your normal terminal prompt.
print("\n[Process Completed Successfully]")