# 
# @author: Allan
#

from tqdm import tqdm
from common import Sentence, Instance
from typing import List
import re


class Reader:

    def __init__(self):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.vocab = set()

    def read_txt(self, file: str, number: int = -1, if_classify: bool = False) -> List[Instance]:
        print("Reading file: " + file)
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            sents = []
            ori_sents = []
            labels = []
            max_length = -1
            curr_length = 0
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "" and len(sents) > 0:
                    insts.append(Instance(Sentence(sents, ori_sents), labels))
                    sents = []
                    ori_sents = []
                    labels = []
                    max_length = max(curr_length, max_length)
                    curr_length = 0
                    if len(insts) == number:
                        break
                    continue
                ls = line.split('\t')
                if not if_classify:
                    label = ls[-1]
                    sent = ls[-2]
                else:
                    sent = ls[-1]
                    label = 'O'
                curr_length += len(sent)
                ori_sents.append(sent)
                sents.append(sent)
                labels.append(label)

        print("number of sentences: {}".format(len(insts)))
        print(f"maximum paragrpah length is: {max_length}")
        return insts



