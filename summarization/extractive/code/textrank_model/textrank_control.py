# from gensim.summarization.summarizer import summarize
from textrank_summarizer import summarize
import time
import sys
from ast import literal_eval
import numpy as np
from tqdm import tqdm

DATA_PATH="../../data/labeled_source/filtered_test_labeled_sentences.txt"
LABEL_PATH="../../data/target/filtered_test_sentence_labels.txt"
RES_PATH = "./textrank_control.result"

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
    
def textrank_summarize(corpus, summary_label_list):
    print("Begin summarizing...")

    list_of_summarization = []

    error_counter = 0
    null_summarization_counter = 0
    for i in tqdm(range(len(corpus))):
        sample = corpus[i]
        label_list = summary_label_list[i]
        sentences = [item[0] for item in sample]
        cmp_length=50
        cmp_sents = [sent[:cmp_length] for sent in sentences]
        labels = [item[-1] for item in sample]
        summarization = summarize(sentences, ratio=1, split = True)

        sorted_ix = []
        for sent in summarization:
            try:
                sorted_ix.append(cmp_sents.index(sent[:cmp_length]))
            except:
                found=False
                for pos in range(len(sentences)):
                    if sent[:cmp_length] in sentences[pos]:
                        sorted_ix.append(pos)
                        found=True
                        break
                if not found:
                    print('error')
        assert len(sorted_ix) == len(summarization)

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
            sorted_ix.remove(idx) 

        list_of_summarization.append(' '.join(control_selected_summary))

    return list_of_summarization

if __name__ =='__main__':
    corpus = get_corpus()
    summary_label_list = get_summary_label_list()
    assert len(corpus) == len(summary_label_list)
    res = textrank_summarize(corpus, summary_label_list)

    with open(RES_PATH, 'w', encoding='utf-8') as fw:
        for sample in res:
            fw.write(sample + "\n")