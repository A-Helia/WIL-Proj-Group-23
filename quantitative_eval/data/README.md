# Walert - A Conversational Agent
Demo Video Link: https://bit.ly/chiir24walertdemovideo

## About Walert
Walert is a conversational agent designed to answer frequently asked questions (FAQs) regarding programs of study offered at the School of Computing Technologies, RMIT University. Our intent-based approach, deployed on Amazon Echo devices, was showcased as a live demo during RMIT University's Open Day in August 2023.


Note: This repository contains all utility code for 'Behind The Scenes' of Walert.

## Dataset Files
In the dataset folder, you'll find the following files that are generated from an FAQ Document:

1. **File Name: topics.csv**\
   -*Format:* CSV\  
   -*Description:* Contains unique query IDs extracted from FAQs, along with variations for each query. Utilizes Falcon for formatting different question formats related to the same query.\  
   -*Columns:*
     - `topic_id`: Unique ID for each query (e.g., W01).
     - `Topic`: Title of the query as it appears in the FAQ (e.g., what's the difference between CS and SE programs?).
     - `question_id`: Unique ID for each question related to a specific topic (e.g., W01Q01).
     - `question`: Alternative question related to each query, including the main format from the FAQ (e.g., What sets apart CS programs from SE programs?).
Total topics: 43, with 1 to 4 alternative questions for each topic.


2. **File Name: Collection.csv**\
   -*Format:* CSV\
   -*Description:* Corpus of passages extracted from FAQ answers, representing the knowledge base (KB).\
   -*Columns:*
     - `passage_id`: Unique identifier for each passage (e.g., P01).
     - `passage`: Exact text passage extracted from the FAQ, forming the answer or part of an answer for a question of a specific topic (e.g., "The first two years of these two programs are near-identical..." ).


3. **File Name: groundtruth.csv**\
   -*Format*: CSV\
   -*Description*: Mapping between unique topics and relevant passages in a one-to-many relationship. Includes a relevance judgment score (1 or 2).\
   -*Columns*:
     - `topic_id`: ID for a query (e.g., W01)
     - `topic`: Title of the query (e.g., what's the difference between CS and SE programs?)
     - `passage_id`: ID for the passage that answers the query (e.g., P01)
     - `passage`: Exact text of the passage that answers the query (e.g., "The first two years of these two programs are near-identical..." ).
     - `relevance_judgment`: Score (1 or 2) indicating full (2) or partial (1) relevance of the passage to the topic.


4. **File Name: qrels.txt**\
   -*Format*: TXT\
   -*Description*: Mapping between each question ID for all topics and its related passage IDs in a one-to-many relationship. Includes relevant scores (1 or 2).\
   -*Example line*: "W01Q01 0 P01 2" means the answer to the first question related to topic 1 is fully answered in passage 1.


5. **File Name: gold_summaries.csv**\
   -*Format*: CSV\
   -*Description*: Includes ideal answers for questions that are partially mapped to one or more passages. Passages are combined and summarized.\
   -*Columns*:
     - `question_id`: ID for a question (e.g., W01Q01).
     - `summary_id`: ID of a summary answer from more than one passage (e.g., S21). In case the question is partially relevant to only one passage summary_id is the ID of that passage (e.g., P01).
     - `summary`: the generated summary of the passages or the passage that answers the question (e.g., "You'll receive credit for only 75% of the courses you take in the first year..."). 
     - `passage_id`: the passage ID where the summary is extracted from (e.g., P61). 


Note: intent_mapping is used for the intent-based system. It will not be used to build or train RAG. 
