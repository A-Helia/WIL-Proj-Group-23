import sounddevice as sd
from scipy.io.wavfile import write
import threading
from pydub import AudioSegment
import numpy as np
import whisper
import os
from pyserini.search import FaissSearcher
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from gtts import gTTS
# from pydub import AudioSegment
import pygame
import time
import os
import logging
import requests
import json

URL = "http://ec2-3-25-226-226.ap-southeast-2.compute.amazonaws.com:8001/get_generatedResponse/"

logging.basicConfig(filename='voice_assistant.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Overall System Pipeline
# [Voice Recording locally] -> {User Query in audio format} -> [ASR using Open AI's Whisper] -> {Query in text format} -> \
# [Dense Retrieval] -> {Top 3 Passages} -> [Textual Response Generation using Falcon] -> {System Response in Text Format} -> [TTS] -> {Final Voice Response in audio format} -> [Play Voice Response using Pygame]

def play_voice_response(text):
    # Language in which you want to convert

    language = 'en'
    # Passing the text and language to the engine
    tts = gTTS(text=text, lang=language, slow=False)

    # Saving the converted audio in a file named 'output.wav'
    tts.save("response.wav")
    # sound = AudioSegment.from_mp3("response.wav")
    # sound.export("response.wav", format="wav")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("response.wav")
    pygame.mixer.music.play()
    time.sleep(3)

    os.remove("response.wav")
    # os.remove("response.mp3")


DATA_DIR = "../../data"

COLLECTION = DATA_DIR + "/collection.csv"
TOPICS = DATA_DIR + "/topics.csv"
GROUNDTRUTH = DATA_DIR + "/groundtruth.csv"

# Dense Retrieval
INDEX = "../../target/indexes/tct_colbert-v2-hnp-msmarco-faiss"
QUERY_ENCODER = 'facebook/dpr-question_encoder-multiset-base'
OUTPUT_PATH = '../../target/runs/rag-dense-faiss.txt'
RUN = "dense-faiss"

searcher = FaissSearcher(
    INDEX,
    QUERY_ENCODER
)


def get_context_passages(question):
    num_hits = 10
    hits = searcher.search(question, num_hits)
    top_K = 3
    collection_df = pd.read_csv(COLLECTION)
    context_passages = []
    for d in hits[:top_K]:
        temp_passage = list(collection_df[collection_df['passage_id'] == d.docid]['passage'])[0]
        context_passages.append(temp_passage)
    return context_passages

# Create a callback function to stop the recording
def callback(indata, frames, time, status):
    print("...")
    audio_data.append(indata.copy())
    if EVENT.is_set():
        print("Recording finished.")
        raise sd.CallbackStop



if __name__ == '__main__':
    logging.info("Starting the voice assistant...")


    play_voice_response('Hi, I am Walert. How may I help you?')
    while True:
        
        
        # ************ Query Recording ************

        samplerate = 44100  # Standard for most microphones
        channels = 2  # Stereo

        audio_data = []

        # Create an event for stopping the recording
        EVENT = threading.Event()

        # Start the recording in a new thread
        stream = sd.InputStream(callback=callback, channels=channels, samplerate=samplerate)
        with stream:
            # Wait for the user to press Enter
            input()
            EVENT.set()
        
        # Concatenate the audio data and save it to a temporary WAV file
        audio = np.concatenate(audio_data)
        temp_filename = 'user_voice_query.wav'
        write(temp_filename, samplerate, audio)

        logging.info("User's voice query successfully recorded and store as user_voice_query.wav")

            # ************ ASR ************
        model = whisper.load_model("base")
        result = model.transcribe(temp_filename, fp16=False)
        question = result["text"]

        logging.info(f"User's voice query trasncribed: {question}")

        logging.info(f"Initiating retrieval . . .")
        # ************ Retrieval ************
        RAG_context_passages = get_context_passages(question)

        logging.info(f"Retrieval Completed")

        # ************ Response Generation in Text ************
        logging.info(f"Initiating response generation using Falcon in-house API endpoint")

        headers = {"Content-Type": "application/json"}

        static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on the retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
        prompt_base = static_prompt + "\n Question: " + question + "\n Document 1: " + RAG_context_passages[0] + "\n Document 2: " + \
                    RAG_context_passages[1] + "\n Document 3: " + RAG_context_passages[2] + '\n Answer: '
        
        prompt_data = {"prompt": prompt_base}
        response = requests.post(URL, headers=headers, data=json.dumps(prompt_data))

        response_text = response.json()['Response']

        logging.info(f"Response generation completed (text format): {response_text}")

        # ************ Voice Response ************
        logging.info(f"Initiating voice response (audio format)")

        play_voice_response(response_text)

        play_voice_response('How else can i help you?')



