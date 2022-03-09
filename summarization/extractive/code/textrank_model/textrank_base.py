from textrank_summarizer import summarize
import time
import sys
from ast import literal_eval
from tqdm import tqdm 

DATA_PATH="../../data/labeled_source/filtered_test_labeled_sentences.txt"
LABEL_PATH="../../data/target/filtered_test_sentence_labels.txt"
RES_PATH = "./textrank_base.result"

# read data
def get_corpus():
    """
        process into list of documents
        each documents contains list of (sent, label)
    """
    corpus = []
    with open(DATA_PATH, 'r', encoding='utf-8') as src:
        documents = src.read().strip().split('\n\n')
        for doc in documents:
            processed_sent_list = []
            lines = doc.strip().split('\n')
            for line in lines:   
                _, sent, label =  line.split('\t')
                processed_sent_list.append(sent) # here only append sent, since we are not using label
            corpus.append(processed_sent_list)
    return corpus

def get_summary_size_list():
    summary_size_list=[]
    with open(LABEL_PATH, 'r', encoding='utf-8') as ctrl:
        for line in ctrl:
            label_list = literal_eval(line.strip().split('\t')[-1])
            summary_size_list.append(len(label_list))
    return summary_size_list
    
def textrank_summarize(corpus, summary_size_list):
    print("Begin summarizing...")

    list_of_summarization = []

    error_counter = 0
    null_summarization_counter = 0
    for i in tqdm(range(len(corpus))):
        sample = corpus[i]
        summary_size = summary_size_list[i]
        ratio = summary_size/len(sample)
        summarization = summarize(sample, ratio=ratio, split = True)

        # fix floating point error that cause fewer sentences to be produced
        while len(summarization) < summary_size_list[i]:
            summary_size+= 1
            ratio = summary_size/len(sample)
            summarization = summarize(sample, ratio=ratio, split = True)

        list_of_summarization.append(' '.join(summarization))

    return list_of_summarization

if __name__ =='__main__':
    corpus = get_corpus()
    summary_size_list = get_summary_size_list()
    print("read corpus number:", len(corpus))
    print("read label number:", len(summary_size_list))
    res = textrank_summarize(corpus, summary_size_list)
    with open(RES_PATH, 'w', encoding='utf-8') as fw:
        for sample in res:
            fw.write(sample + "\n")	