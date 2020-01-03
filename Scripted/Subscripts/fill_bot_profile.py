import pickle
import collections

def process():
    with open("Processed_Scripts/intent_query_script.pkl",'rb') as fp:
        intent_query_dict = pickle.load(fp)

    with open("Processed_Scripts/intent_response_script.pkl",'rb') as fp:
        intent_response_dict = pickle.load(fp)

    qa_ordered_dict = collections.OrderedDict()

    for intent in intent_response_dict:
        queries = intent_query_dict[intent]
        for query in queries:
            qa_ordered_dict[query] = intent_response_dict[intent]


    with open("Processed_Scripts/Bot_Profile.pkl", "wb") as fp:
        pickle.dump(qa_ordered_dict, fp)
