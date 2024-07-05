import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import logging
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from pynput.keyboard import Controller, Key
import threading

# Configure logging
logging.basicConfig(format='%(asctime)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global variables
keyboard = Controller()
is_typing = False

class VoiceTyper:
    def __init__(self, args):
        self.args = args
        self.data_queue = Queue()
        self.recorder = self.setup_recorder()
        self.audio_model = self.load_audio_model()
        self.transcription = ['']
        self.phrase_time = None

    def setup_recorder(self):
        recorder = sr.Recognizer()
        recorder.energy_threshold = self.args.energy_threshold
        recorder.dynamic_energy_threshold = False
        return recorder

    def load_audio_model(self):
        model = self.args.model
        if self.args.model != "large" and not self.args.non_english:
            model = model + ".en"
        return whisper.load_model(model)

    def record_callback(self, _, audio):
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def type_text(self, text):
        for letter in text:
            if not is_typing:
                break
            self.keyboard.type(letter)
            sleep(0.01)

    def process_audio(self):
        now = datetime.utcnow()
        if self.data_queue.empty():
            return

        phrase_complete = self.check_phrase_complete(now)
        self.phrase_time = now

        audio_data = b''.join(self.data_queue.queue)
        self.data_queue.queue.clear()

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        result = self.transcribe_audio(audio_np)
        text = result['text'].strip()

        self.update_transcription(text, phrase_complete)
        self.print_transcription()
        
        if is_typing:
            threading.Thread(target=self.type_text, args=(text,)).start()
        self.check_for_commands(text)

    def check_phrase_complete(self, now):
        return self.phrase_time and now - self.phrase_time > timedelta(seconds=self.args.phrase_timeout)

    def transcribe_audio(self, audio_np):
        extra_args = {"word_timestamps": True}
        if not self.args.non_english:
            return self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), **extra_args)
        elif self.args.language:
            return self.audio_model.transcribe(audio_np, language=self.args.language, fp16=torch.cuda.is_available(), **extra_args)
        else:
            return self.audio_model.transcribe(audio_np, language="no", fp16=torch.cuda.is_available(), **extra_args, prepend_punctuations=True)

    def update_transcription(self, text, phrase_complete):
        if phrase_complete:
            self.transcription.append(text)
        else:
            self.transcription[-1] = text

    def print_transcription(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        for line in self.transcription:
            print(line)
        print('', end='', flush=True)

    def check_for_commands(self, text):
        text_lower = text.lower()
        commands = {
            "type start": self.start_typing,
            "type stop": self.stop_typing,
            # Add more commands here
        }
        
        for command, action in commands.items():
            if command in text_lower or all(word in text_lower for word in command.split()):
                logger.info(f">>>{command}")
                action()
                return
        
        logger.info(">>>no valid action")

    def start_typing(self):
        global is_typing
        is_typing = True

    def stop_typing(self):
        global is_typing
        is_typing = False

    def run(self):
        with sr.Microphone(sample_rate=16000) as source:
            self.recorder.adjust_for_ambient_noise(source)
            self.recorder.listen_in_background(source, self.record_callback, phrase_time_limit=self.args.record_timeout)
            
            print("Model loaded.\n")
            
            while True:
                try:
                    self.process_audio()
                    sleep(0.25)
                except KeyboardInterrupt:
                    break

        print("\n\nTranscription:")
        for line in self.transcription:
            print(line)

def main():
    parser = argparse.ArgumentParser()
    # Add your existing arguments here
    args = parser.parse_args()

    voice_typer = VoiceTyper(args)
    voice_typer.run()

if __name__ == "__main__":
    main()