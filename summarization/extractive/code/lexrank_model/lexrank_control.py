from lexrank import STOPWORDS, LexRank
import time
import sys
from ast import literal_eval
import numpy as np

DATA_PATH="../../data/labeled_source/filtered_test_labeled_sentences.txt"
LABEL_PATH="../../data/target/filtered_test_sentence_labels.txt"
RES_PATH = "./lexrank_control.result"

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
                # sent = sent.replace(" <sep> "," ").replace(" <REVBREAK> ", " ").strip()  # cleanup
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

def lexrank_summarize(corpus, summary_label_list):
    list_of_summarization = []
    # documents: [[sent1,sent2,...],[sent1,sent2, ...]] list of document of list of sents in each document
    documents = [ [item[0] for item in doc] for doc in corpus ]
    print("[" + "Document Size: " + str(len(documents)) + "]")
    print("[" + time.strftime("%H:%M:%S", time.localtime()) + "]", "Begin building LexRank model...")	
    lxr = LexRank(documents, stopwords=STOPWORDS['en'])
    print("[" + time.strftime("%H:%M:%S", time.localtime()) + "]", "LexRank model successfully built...")

    for i in range(len(corpus)):
        sample = corpus[i] # list of (sent,label) in the document
        label_list = summary_label_list[i]
        sentences = [item[0] for item in sample]
        lex_scores = lxr.rank_sentences(sentences)
        sorted_ix = np.argsort(lex_scores)[::-1]

        control_selected_summary = []
        for label in label_list:
            selected_sent = None # (sent, label)
            selected_id = -1
            for idx in sorted_ix:
                if sample[idx][-1] == label:
                    selected_sent = sample[idx]
                    selected_id = idx
                    break
            if selected_sent is None:
                # find the first sentence in sorted_ix that don't have label in label_list
                dup_label_list = label_list.copy() # for selection later, mutable
                for idx in sorted_ix:
                    if sample[idx][-1] not in dup_label_list:
                        selected_sent = sample[idx]
                        selected_id = idx
                        break
                    else:
                        dup_label_list.remove(sample[idx][-1])
            control_selected_summary.append(selected_sent[0])
            # to avoid repetition of selection, remove first occurrence
            sorted_ix = np.delete(sorted_ix, np.where(sorted_ix == idx)[0])
            # sorted_ix.remove(idx) 
        list_of_summarization.append(" ".join(control_selected_summary))

    return list_of_summarization

if __name__ =='__main__':
    corpus = get_corpus()
    summary_label_list = get_summary_label_list()
    assert len(corpus) == len(summary_label_list)
    res = lexrank_summarize(corpus, summary_label_list)

    with open(RES_PATH, 'w', encoding='utf-8') as fw:
        for sample in res:
            fw.write(sample + "\n")