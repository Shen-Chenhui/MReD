
from typing import List, Dict, Tuple
from common import Instance
import torch
from collections import defaultdict, Counter

class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))


def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.Tensor,
                         batch_gold_ids: torch.Tensor,
                         word_seq_lens: torch.Tensor,
                         idx2label: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    batch_p_dict = defaultdict(int)
    batch_total_entity_dict = defaultdict(int)
    batch_total_predict_dict = defaultdict(int)

    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction

        #convert to span
        output_spans = set()
        for i in range(len(output)):
            if output[i].startswith('O'):
                output_spans.add(Span(i,i,output[i][0]))
            else:
                output_spans.add(Span(i,i,output[i][2:]))
                batch_total_entity_dict["1"]+=1
            #if output[i].startswith("B-"):
            #    output_spans.add(Span(i, i, "1"))
            #    batch_total_entity_dict["1"] += 1
            #if output[i].startswith("E-"):
            #    for j in range(len(output)):
            #        if output[j].startswith("C-") and output[i][2:] == output[j][2:]:
            #            output_spans.add(Span(i, j, "1"))
            #            batch_total_entity_dict["1"] += 1
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith('O'):
                predict_spans.add(Span(i,i, prediction[i][0]))
            else:
                predict_spans.add(Span(i,i,prediction[i][2:]))
                batch_total_predict_dict["1"]+=1
            #if prediction[i].startswith("B-"):
            #    predict_spans.add(Span(i, i, "1"))
            #    batch_total_predict_dict["1"] += 1
            #if prediction[i].startswith("E-"):
            #    for j in range(len(prediction)):
            #        if prediction[j].startswith("C-") and prediction[i][2:] == prediction[j][2:]:
            #            predict_spans.add(Span(i, j, "1"))
            #            batch_total_predict_dict["1"] += 1

        correct_spans = predict_spans.intersection(output_spans)
        for span in correct_spans:
            batch_p_dict["1"] += 1

        # for pred, gold in zip(output, prediction):
        #     if pred != "O":
        #         batch_total_predict_dict["tok"] += 1
        #     if gold != "O":
        #         batch_total_entity_dict["tok"] += 1
        #     if pred == gold and pred != "O":
        #         batch_p_dict["tok"] += 1

    return Counter(batch_p_dict), Counter(batch_total_predict_dict), Counter(batch_total_entity_dict)
