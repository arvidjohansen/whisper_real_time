#! python3.7

import argparse

import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta

from queue import Queue
from time import sleep
from sys import platform
import logging


logging.basicConfig(format='%(asctime)s %(message)s',
                             datefmt='%m/%d/%Y %I:%M:%S %p',
                             #level=logging.DEBUG,
                             )      
logger = logging.getLogger(__name__)    
logger.setLevel(logging.DEBUG)  

from pynput.keyboard import Controller, Key
import threading

keyboard = Controller()
is_typing = False

def type_text(letter=''):
    global is_typing
    
    while is_typing:
        if letter:
            keyboard.type(letter)
            sleep(1)  # Delay between each keypress
def type_letter(letter):
    keyboard.type(letter)
    sleep(0.01)  # Delay between each keypress
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "norwegian"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--language", default = 'en',
                        help="Specify which language to use.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=1,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retrieved from the queue.
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    #elif args.model != "large" and args.non_english:
    #    model = "pytorch_model.bin"
        
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                extra_args = {
                    "word_timestamps":True,
                    
                              }
                if not args.non_english:
                    result = audio_model.transcribe(audio_np, 
                                                    fp16=torch.cuda.is_available(),
                                                    **extra_args)
                elif args.language:
                    result = audio_model.transcribe(audio_np, 
                                                    language=args.language, 
                                                    fp16=torch.cuda.is_available(),
                                                    **extra_args)
                else:
                    result = audio_model.transcribe(audio_np, 
                                                    language="no", 
                                                    fp16=torch.cuda.is_available(),
                                                   **extra_args,prepend_punctuations=True)
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    # Phrase is complete. 
                    # First check for commands
                    # If type start, enable function but DONT type it
                    # If type stop, disable function but DONT type it
                    # If insert symbol, insert correct symbol
                    

                    cmd_name = check_for_commands(text)
                    if cmd_name:
                        logger.debug(f'Preparing execution of command: {cmd_name}')
                        text_to_insert = run_command(cmd_name)
                        if text_to_insert:
                            transcription.append(text_to_insert)


                    transcription.append(text)

                    # actually print the characters (if enabled)
                    global is_typing
                    if is_typing:
                        for letter in text:
                            type_letter(letter)

                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for line in transcription:
                    print(line)
                    

                # Flush stdout.
                print('', end='', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)
# Add new supported commands here
# Format: Tuple
# (trigger_words(list:str), command_name(str))

SUPPORTED_COMMANDS = [
    (['type start','typestart','starttype','start type'],'type_start'),
    (['type stop','typestop','stoptype','stop type'],'type_stop'),
    (['new line','newline','insert line','insertline'],'insert_line'),
]
def check_for_commands(text:str):
    """This function takes the inputted text,
    loops through every trigger_word combination 
    defined in SUPPORTED_COMMANDS
    and checks for all possible matches.

    Will automatically call the associated command.

    Keyword arguments:
    text -- str (the input text that has just been translated)
    Return: str (name of matched command, otherwise empty string '')
    """
    
    text = text.lower()
    logger.debug(f'Checking for command in following text: "{text}"')
    
    # Loop through supported commands
    for trigger_word_list,command_name  in SUPPORTED_COMMANDS:
        # Loop through every combination in the first part of the tuple
        for combo in trigger_word_list:
            # Check every combination for match
            logger.debug(f'Checking for match with "{combo}"')
            if combo == text:
                # Match successfull
                logger.debug(f'Success! Executing command: {command_name} ')
                return command_name
    return ''
    




    """
    if "type start" in text_lower or "typestart" in text_lower or "type" in text_lower and "start" in text_lower:
        logger.info(">>>type start")
        run_command("type start")
    elif "type stop" in text_lower or "typestop" in text_lower or "type" in text_lower and "stop" in text_lower:
        run_command("type stop")
        logger.info(">>>type stop") 
    else:
        logger.info(">>>no valid action")
    """

def run_command(command):
    global is_typing
    if command == "type_start":
        is_typing = True
    elif command == "type_stop":
        is_typing = False
    elif command == "insert_line":
        return '\n'

if __name__ == "__main__":
    main()
