import time
import sys
sys.path.insert(1, '../textrank_model')
from textrank_summarizer import summarize
from ast import literal_eval
import numpy as np
from tqdm import tqdm
import torch 
from collections import Counter


# read data
def get_corpus(truncation=None):
    """
        process into list of documents
        each documents contains list of (sent, label)
    """
    corpus = []
    with open(DATA_PATH, 'r', encoding='utf-8') as src:
        documents = src.read().strip().split('\n\n')
        if truncation is not None: # process the documents in chunks
            start, end = truncation
            documents = documents[start:end]
        for doc in documents:
            processed_sent_list = []
            lines = doc.strip().split('\n')
            for line in lines:  
                _, sent, label =  line.split('\t')
                sent = sent.replace(" <sep> "," ").replace(" <REVBREAK> ", " ").strip()  # cleanup        
                processed_sent_list.append((sent,label))
            corpus.append(processed_sent_list)
    return corpus


def get_summary_label_list():
    summary_label_list=[]
    with open(LABEL_PATH, 'r', encoding='utf-8') as ctrl:
        for line in ctrl:
            label_list = literal_eval(line.strip().split('\t')[-1])
            summary_label_list.append(label_list)
    return summary_label_list
    
def textrank_summarize(corpus, summary_label_list, max_label_sents):
    print("Begin summarizing...")

    list_of_summarization = []
    labelled_sent_dict = {}
    # store all sentences according to their label groups
    for i in range(len(corpus)):
        sample = corpus[i]
        for sent, label in sample:
            if label in labelled_sent_dict:
                labelled_sent_dict[label].append(sent)
            else:
                labelled_sent_dict[label] = [sent]
    # get top n sentences
    top_labelled_sents = {}
    for label in tqdm(labelled_sent_dict.keys()):
        # below +1 to ensure at least enough required sentences are extracted
        ratio = min((max_label_sents[label]+1) / len(labelled_sent_dict[label]), 1) # ratio <= 1
        top_labelled_sents[label] = summarize(labelled_sent_dict[label], ratio=ratio, split = True)
    torch.save(top_labelled_sents, TORCH_PATH)

    if truncation is not None:
        return []

    # comb into meta-reviews by using most typical labells
    for label_list in summary_label_list:
        pointer = {} # keep track of used top sentences so no repetition
        for label in top_labelled_sents.keys():
            pointer[label] = 0 
        control_selected_summary = []
        for label in label_list:
            pos = pointer[label]
            pointer[label] = pointer[label]+1 # increment pointer
            if pos == len(top_labelled_sents[label]): 
                # print("exceeding, need to increase n for: ", label)
                pos -= 1 
            control_selected_summary.append(top_labelled_sents[label][pos])
        list_of_summarization.append(' '.join(control_selected_summary))

    return list_of_summarization

if __name__ =='__main__':
    if len(sys.argv) == 5:
        # if takes too long (due to too much training data), process in chunks 
        # USAGE: python generic.py <dataset_path> <label_path> <chunk_id>
        DATA_PATH=sys.argv[1]
        LABEL_PATH = sys.argv[2]
        chunk_id = sys.argv[3]
        RES_NAME = sys.argv[4]
        start = int(chunk_id) * 600
        end = (int(chunk_id)+1) * 600
        truncation = (start,end)
        print("processing for documents:", truncation, " for id:", chunk_id)
        RES_PATH = RES_NAME+'_'+str(chunk_id)+'.result'
        TORCH_PATH = RES_NAME+'_'+str(chunk_id)+'.pt'
        print("chunk processed item will be saved to:", TORCH_PATH)
    elif len(sys.argv) == 4:
        # for low score and high score papers, running this is sufficient
        DATA_PATH=sys.argv[1]
        LABEL_PATH = sys.argv[2]
        RES_NAME = sys.argv[3]
        truncation = None
        RES_PATH = RES_NAME+'.result'
        TORCH_PATH = RES_NAME+'.pt'
        print("result will be written to:", RES_PATH)
    else:
        print("[USAGE]: python generic.py <dataset_path> <label_path> <name: eg. src_generic, src_low_score, src_high_score, tgt_generic, tgt_low_socre, tgt_high_score>")
        exit()
    corpus = get_corpus(truncation)
    summary_label_list = get_summary_label_list()

    # ONLY classify the max number of sents for each label required
    max_label_sents = {}
    for item in summary_label_list:
        tmp = Counter(item)
        for label in tmp.keys():
            if label in max_label_sents:
                max_label_sents[label] = max(max_label_sents[label], tmp[label])
            else:
                max_label_sents[label] = tmp[label]
    print("metareview max label sents:", max_label_sents)
    res = textrank_summarize(corpus, summary_label_list, max_label_sents)
    
    if truncation is None:
        with open(RES_PATH, 'w', encoding='utf-8') as fw:
            for sample in res:
                fw.write(sample + "\n")