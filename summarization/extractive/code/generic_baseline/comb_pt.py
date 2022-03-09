import torch
import sys
sys.path.insert(1, '../textrank_model')
from textrank_summarizer import summarize
from ast import literal_eval
from tqdm import tqdm
from collections import Counter

def textrank_summarize(labelled_sent_dict, summary_label_list, max_label_sents):
    print("Begin summarizing...")
    list_of_summarization = []
    top_labelled_sents = {}
    # get top n sentences
    for label in tqdm(labelled_sent_dict.keys()):
        # below +1 to ensure at least enough required sentences are extracted
        ratio = min((max_label_sents[label]+1) / len(labelled_sent_dict[label]), 1) # ratio <= 1
        top_labelled_sents[label] = summarize(labelled_sent_dict[label], ratio=ratio, split = True)
    if 'O' in top_labelled_sents:
        top_labelled_sents['misc'] = top_labelled_sents['O']
    torch.save(top_labelled_sents, TORCH_PATH)

    # comb into meta-reviews by using most typical labells
    for label_list in summary_label_list:
        pointer = {} # keep track of used top sentences so no repetition
        for label in top_labelled_sents.keys():
            pointer[label] = 0 
        # pointer['O'] = 0    # for src
        # pointer['misc'] = 0    # for src
        control_selected_summary = []
        for label in label_list:
            pos = pointer[label]
            pointer[label] = pointer[label]+1 # increment pointer
            # if pos == len(top_labelled_sents[label]): # check if n = 10 is enough
            #     print("exceeding, need to increase n for: ", label)
            #     pos -= 1 
            control_selected_summary.append(top_labelled_sents[label][pos])
        list_of_summarization.append(' '.join(control_selected_summary))

    return list_of_summarization


def get_summary_label_list():
    summary_label_list=[]
    with open(LABEL_PATH, 'r', encoding='utf-8') as ctrl:
        for line in ctrl:
            label_list = literal_eval(line.strip().split('\t')[-1])
            summary_label_list.append(label_list)
    return summary_label_list

def get_labelled_sent_dict(n):
    prefiltered_sents = {}
    for file_id in range(n):       
        tmp_path = RES_NAME+'_'+str(file_id)+'.pt'
        tmp = torch.load(tmp_path)
        for label in tmp:
            if label in prefiltered_sents:
                prefiltered_sents[label] = prefiltered_sents[label] + tmp[label]
            else:
                prefiltered_sents[label] = tmp[label]
        print("processed: ", tmp_path)
    return prefiltered_sents

if __name__ =='__main__':
    if len(sys.argv) == 4:
        LABEL_PATH = sys.argv[1]
        TOTOAL_CHUNKS = int(sys.argv[2])
        RES_NAME = sys.argv[3]
        RES_PATH = RES_NAME+'.result'
        TORCH_PATH = RES_NAME+'.pt'
        print("result will be written to:", RES_PATH)
    else:
        print("[USAGE]: python generic.py <label_path> <total chunks> <name: eg. src_generic, src_low_score, src_high_score, tgt_generic, tgt_low_socre, tgt_high_score>")
        exit()

    summary_label_list = get_summary_label_list()

    # use max number of sents for each label in all meta-reviews
    max_label_sents = {}
    for item in summary_label_list:
        tmp = Counter(item)
        for label in tmp.keys():
            if label in max_label_sents:
                max_label_sents[label] = max(max_label_sents[label], tmp[label])
            else:
                max_label_sents[label] = tmp[label]
    print("metareview max label sents:", max_label_sents)
    max_label_sents['O'] = max_label_sents['misc'] # fix key error
 
    labelled_sent_dict = get_labelled_sent_dict(TOTOAL_CHUNKS)

    res = textrank_summarize(labelled_sent_dict, summary_label_list, max_label_sents)
    
    with open(RES_PATH, 'w', encoding='utf-8') as fw:
        for sample in res:
            fw.write(sample + "\n")