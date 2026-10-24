from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model=AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

def gen_answer_base1_v1(row):
  static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on the retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
  prompt_base1 = static_prompt + "\nQuestion: "+row['question_content'] + "\nDocument 1: "+ row['top1_content'] + '\nAnswer: '
  gen_answer = pipeline(
      prompt_base1,
      eos_token_id=tokenizer.eos_token_id,
      do_sample=False,
      #max_length=800,
      max_new_tokens=100,
      #top_k=2,
      #max_new_tokens=400,
      top_k=10,
      top_p=0.95,
      typical_p=0.95,
      temperature=0.0,
      repetition_penalty=1.03,
  )
  return gen_answer



def gen_answer_base3_v1(row):
  static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on 3 retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
  prompt_base3 = static_prompt + "\nQuestion: "+row['question_content'] + "\nDocument 1: "+ row['top3_content'][0] + "\nDocument 2: "+row['top3_content'][1]+"\nDocument 3: "+row['top3_content'][2]+'\nAnswer: '
  gen_answer = pipeline(
      prompt_base3,
      eos_token_id=tokenizer.eos_token_id,
      do_sample=False,
      #max_length=800,
      max_new_tokens=100,
      #top_k=2,
      #max_new_tokens=400,
      top_k=10,
      top_p=0.95,
      typical_p=0.95,
      temperature=0.0,
      repetition_penalty=1.03,
  )
  return gen_answer



def gen_answer_base5_v1(row):
  static_prompt = "Generate an answer to be synthesized with text-to-speech for a virtual assisstant, the answer should be based on 5 retrieved documents for the following question. If the retrieved documents are not related to the question, then answer NA."
  prompt_base5 = static_prompt + "\nQuestion: "+row['question_content'] + "\nDocument 1: "+ row['top5_content'][0] + "\nDocument 2: "+row['top5_content'][1]+"\nDocument 3: "+row['top5_content'][2]+"\nDocument 4: "+row['top5_content'][3]+"\nDocument 5: "+row['top5_content'][4]+'\nAnswer: '
  gen_answer = pipeline(
      prompt_base5,
      eos_token_id=tokenizer.eos_token_id,
      do_sample=False,
      #max_length=800,
      max_new_tokens=100,
      #top_k=2,
      #max_new_tokens=400,
      top_k=10,
      top_p=0.95,
      typical_p=0.95,
      temperature=0.0,
      repetition_penalty=1.03,
  )
  return gen_answer
