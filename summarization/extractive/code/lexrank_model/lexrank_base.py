from lexrank import STOPWORDS, LexRank
import time
import sys
from ast import literal_eval

DATA_PATH="../../data/labeled_source/filtered_test_labeled_sentences.txt"
LABEL_PATH="../../data/target/filtered_test_sentence_labels.txt"
RES_PATH = "./lexrank_base.result"


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
                sent = sent.strip() 
                processed_sent_list.append(sent) 
                # here only append sent, since we are not using label
            corpus.append(processed_sent_list)
    return corpus

def get_summary_size_list():
    summary_size_list=[]
    with open(LABEL_PATH, 'r', encoding='utf-8') as ctrl:
        for line in ctrl:
            label_list = literal_eval(line.strip().split('\t')[-1])
            summary_size_list.append(len(label_list))
    return summary_size_list

def lexrank_summarize(corpus, summary_size_list):
    list_of_summarization = []
    # documents: [[sent1,sent2,...],[sent1,sent2, ...]] list of document of list of sents in each document
    documents = [ [sent for sent in doc] for doc in corpus ]
    print("[" + "Document Size: " + str(len(documents)) + "]")
    print("[" + time.strftime("%H:%M:%S", time.localtime()) + "]", "Begin building LexRank model...")	
    lxr = LexRank(documents, stopwords=STOPWORDS['en'])
    print("[" + time.strftime("%H:%M:%S", time.localtime()) + "]", "LexRank model successfully built...")

    for i in range(len(documents)):
        sample = documents[i] # list of sents in the document
        summary_size = summary_size_list[i]
        summary = lxr.get_summary(sample, summary_size=summary_size)
        list_of_summarization.append(" ".join(summary))

    return list_of_summarization

if __name__ =='__main__':
    corpus = get_corpus()
    summary_size_list = get_summary_size_list()
    assert len(corpus) == len(summary_size_list)
    res = lexrank_summarize(corpus, summary_size_list)
    with open(RES_PATH, 'w', encoding='utf-8') as fw:
        for sample in res:
            fw.write(sample + "\n")