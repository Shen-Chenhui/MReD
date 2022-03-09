# 
# @author: Allan
#

import torch
import torch.nn as nn

from model.module.bilstm_encoder import BiLSTMEncoder
from model.module.linear_crf_inferencer import LinearCRF
from model.module.linear_encoder import LinearEncoder
from model.embedder import TransformersEmbedder
from typing import Tuple
from overrides import overrides


class TransformersCRF(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(TransformersCRF, self).__init__()
        self.embedder = TransformersEmbedder(config, print_info=print_info)
        self.device = config.device
        if config.hidden_dim > 0:
            self.encoder = BiLSTMEncoder(config, self.embedder.get_output_dim(), print_info=print_info)
        else:
            self.encoder = LinearEncoder(config, self.embedder.get_output_dim(), print_info=print_info)
        self.inferencer = LinearCRF(config, print_info=print_info)

    # @overrides
    def forward(self, words: torch.Tensor,
                    num_sents: torch.Tensor,
                    input_mask: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        words = words.to(self.device)
        num_sents = num_sents.to(self.device)
        input_mask = input_mask.to(self.device)
        labels = labels.to(self.device)
        word_rep = self.embedder(words, input_mask)
        lstm_scores = self.encoder(word_rep, num_sents)
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, num_sents.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, num_sents, labels, mask)
        return unlabed_score - labeled_score

    def decode(self, words: torch.Tensor,
                    num_sents: torch.Tensor,
                    input_mask,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        words = words.to(self.device)
        num_sents = num_sents.to(self.device)
        input_mask = input_mask.to(self.device)
        word_rep = self.embedder(words, input_mask)
        features = self.encoder(word_rep, num_sents)
        bestScores, decodeIdx = self.inferencer.decode(features, num_sents)
        return bestScores, decodeIdx
