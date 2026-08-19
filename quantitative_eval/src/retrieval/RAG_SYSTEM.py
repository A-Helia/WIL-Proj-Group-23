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
from pydub import AudioSegment
import pygame
import time
import os
import logging

logging.basicConfig(filename='voice_assistant.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Overall System Pipeline
# [Voice Recording locally] -> {User Query in audio format} -> [ASR using Open AI's Whisper] -> {Query in text format} -> \
# [Dense Retrieval] -> {Top 3 Passages} -> [Textual Response Generation using Falcon] -> {System Response in Text Format} -> [TTS] -> {Final Voice Response in audio format} -> [Play Voice Response using Pygame]


def play_voice_response(text):
    # Language in which you want to convert

    language = 'en'
    # Passing the text and language to the engine
    tts = gTTS(text=text, lang=language, slow=False)

    # Saving the converted audio in a file named 'output.mp3'
    tts.save("response.mp3")
    sound = AudioSegment.from_mp3("response.mp3")
    sound.export("response.wav", format="wav")

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load("response.wav")
    pygame.mixer.music.play()
    time.sleep(5)

    os.remove("response.wav")
    os.remove("response.mp3")

def get_answer(text):
    # Find the index of 'Answer:'
    answer = ''
    index = text.find('Answer:')
    if index != -1:
        # Extract the text after 'Answer:'
        answer_text = text[index + len('Answer:'):]
        answer = answer_text.strip()

    else:
        answer = "I apologize, I have no knowledge about that"

    return answer


def load_model(model_id):
    model_id = "tiiuae/falcon-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto")

    return pipeline, tokenizer


model_name = "tiiuae/falcon-7b-instruct"
PIPELINE, TOKENIZER = load_model(model_name)

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


def generate_answer(question, context, pipeline, tokenizer):
    static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on the retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
    prompt_base = static_prompt + "\n Question: " + question + "\n Document 1: " + context[0] + "\n Document 2: " + \
                  context[1] + "\n Document 3: " + context[2] + '\n Answer: '

    gen_answer = pipeline(
        prompt_base,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        # max_length=800,
        max_new_tokens=100,
        # top_k=2,
        # max_new_tokens=400,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.0,
        repetition_penalty=1.03)

    return gen_answer


# Create a callback function to stop the recording
def callback(indata, frames, time, status):
    print("...")
    audio_data.append(indata.copy())
    if EVENT.is_set():
        print("Recording finished.")
        raise sd.CallbackStop

# Create an event for stopping the recording
EVENT = threading.Event()

if __name__ == '__main__':
    logging.info("Starting the voice assistant...")
    # ************ Query Recording ************

    samplerate = 44100  # Standard for most microphones
    channels = 2  # Stereo

    audio_data = []

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
    logging.info(f"Initiating response generation using Falcon")
    llm_result = generate_answer(question, RAG_context_passages, PIPELINE, TOKENIZER)

    response_text = get_answer(llm_result[0]['generated_text'])

    logging.info(f"Response generation completed (text format): {response_text}")
    # ************ Voice Response ************
    logging.info(f"Initiating voice response (audio format)")
    play_voice_response(response_text)

    logging.info(f"Conversation turn completed.")


