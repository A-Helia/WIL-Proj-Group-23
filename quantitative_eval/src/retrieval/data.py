import pandas as pd

DATA_DIR = "../../data"

COLLECTION = DATA_DIR + "/collection.csv"
TOPICS = DATA_DIR + "/topics.csv"
GROUNDTRUTH = DATA_DIR + "/groundtruth.csv"
INTENT_MAPPING = DATA_DIR + "/intent_mapping.csv"
WALERT_INTENT = DATA_DIR + "/walert_intent_results.csv"

OUTPUT_PATH = '../../target/runs/walert-intent.txt'


def create_qrels(topics_filename, groundtruth_filename):
    topics = pd.read_csv(topics_filename)
    groundtruth = pd.read_csv(groundtruth_filename)
    
    data = pd.merge(topics, groundtruth, on='topic_id')
    

    data['subtopic'] = 0
    print(data.shape)

    qrels=data[['question_id', 'subtopic', 'passage_id', 'relevance_judgment']]
    print(qrels.head())
    qrels.to_csv(DATA_DIR + "/qrels.txt", sep='\t', index=False, header=False)


def create_pyserini_collection(collection_filename):
     collection = pd.read_csv(collection_filename)
     collection.columns = ['id', 'contents']
     collection.to_json(DATA_DIR + "/collection.jsonl", orient='records', lines=True)   

def create_topics_msmarco_format(topics_filename):
    topics = pd.read_csv(topics_filename)  
    topics = topics[['question_id', 'question']]
    topics.to_csv(DATA_DIR + "/topics.msmarco-format.txt", sep='\t', index=False, header=False)


def parse_walert_run(topics_filename, groundtruth_filename, walert_filename, intent_mapping_filename, collection_filename):
    topics = pd.read_csv(topics_filename)
    groundtruth = pd.read_csv(groundtruth_filename)
    intents = pd.read_csv(intent_mapping_filename)
    collection = pd.read_csv(collection_filename)

    merged = pd.merge(topics, groundtruth, on='topic_id')
    merged_intents = pd.merge(merged, intents, on='question')
    pd.set_option('display.max_columns', None)

    #print(merged_intents.head())
    walert = pd.read_csv(walert_filename)
    walert = pd.merge(topics, walert, on="question")

    runid = "walert_intent"  
    
    with open(OUTPUT_PATH, 'w') as output_writer:
        
        for question_id, question in topics[['question_id','question']].values:
            row = walert[walert['question_id'] == question_id]
            if (row.values.shape[0] == 0):
                "No results found for question: {}".format(question)
                continue
        
            intent = row['actual'].values[0]
            
            if intent != "AMAZON.FallbackIntent":
                #obtain the passages associated with the intent
                passages = merged_intents[merged_intents['intent'] == intent]
                
                if (intent == "Summary"):
                    # create a dummy passage:
                    result = "P_Summary"
                elif (intent == "BTS"):
                    result = "P_BTS"
                elif (intent == "Degree_Type"):
                    result = "P_Degree_Type"
                elif (intent == "Comparison_Bachelors_Associate"):
                    result = "P_Comparison_Bachelors_Associate"
                elif (passages.shape[0] == 0):
                    print("No passages found for intent: {}".format(intent))   
                    #print(row)
                    print(intent)
                else:
                    #pick one and return it as the ranking for the intent
                    result = passages.passage_id.values[0]
                #generate line for TREC format:
                line = "{} Q0 {} 1 1.0 {}\n".format(question_id, result, runid)
                output_writer.write(line)



    
    
    
    

#TODO: Create main funcion with argparse
if __name__ == "__main__":
    create_qrels(TOPICS, GROUNDTRUTH)
    create_pyserini_collection(COLLECTION)
    parse_walert_run(TOPICS, GROUNDTRUTH, WALERT_INTENT, INTENT_MAPPING,COLLECTION)




#   create_topics_msmarco_format(TOPICS)

