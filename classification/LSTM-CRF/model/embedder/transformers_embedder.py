import torch
import torch.nn as nn
from config.transformers_util import context_models

from termcolor import colored
class TransformersEmbedder(nn.Module):
    """
    Encode the input with transformers model such as
    BERT, Roberta, and so on.
    """

    def __init__(self, config, print_info=True):
        super(TransformersEmbedder, self).__init__()
        output_hidden_states = False ## to use all hidden states or not
        print(colored(f"[Model Info] Loading pretrained language model {config.embedder_type}", "red"))
        
        try:
            print("loading stored model...")
            self.model = context_models[config.embedder_type]["model"].from_pretrained('/home/admin/workspace/utils/huggingface_models/'+ 
                                                                        config.embedder_type,output_hidden_states= output_hidden_states, return_dict=False)
        except:
            print("no models stored. start downloading...")
            self.model = context_models[config.embedder_type]["model"].from_pretrained(config.embedder_type,
                                                                        output_hidden_states= output_hidden_states, return_dict=False)
        self.parallel = config.parallel_embedder
        if config.parallel_embedder:
            self.model = nn.DataParallel(self.model)
        """
        use the following line if you want to freeze the model, 
        but don't forget also exclude the parameters in the optimizer
        """
        # self.model.requires_grad = False

    def get_output_dim(self):
        ## use differnet model may have different attribute
        ## for example, if you are using GPT, it should be self.model.config.n_embd
        ## Check out https://huggingface.co/transformers/model_doc/gpt.html
        ## But you can directly write it as 768 as well.
        return self.model.config.hidden_size if not self.parallel else self.model.module.config.hidden_size

    def forward(self, word_seq_tensor: torch.Tensor,
                        input_mask: torch.LongTensor) -> torch.Tensor:
        """

        :param word_seq_tensor: (batch_size x num_sents x max_sent_len)
        :param input_mask: (batch_size x num_sents x max_sent_len)
        :return: (batch_size x num_sents x representation)
        """
        batch_size, num_sents, max_sent_len = word_seq_tensor.size()
        word_seq_tensor = word_seq_tensor.view(-1, word_seq_tensor.size(-1))
        input_mask = input_mask.view(-1, input_mask.size(-1))
        # SCH: fix - limit to 512 to ensure not exceeding max pos encoding for roberta-large
        word_seq_tensor = word_seq_tensor[:,:512]
        input_mask = input_mask[:,:512]
        word_rep, _ = self.model(**{"input_ids": word_seq_tensor, "attention_mask": input_mask})
        ## (batch_size x num_sents ,  max_sent_len, hidden_size)
        word_rep = word_rep[:, 0, :].view(batch_size, num_sents, word_rep.size(-1))
        ## (batch_size, num_sents , hidden_size)
        return word_rep
        # ##exclude the [CLS] and [SEP] token
        # # _, _, word_rep = self.model(**{"input_ids": word_seq_tensor, "attention_mask": input_mask})
        # # word_rep = torch.cat(word_rep[-4:], dim=2)
        # batch_size, _, rep_size = word_rep.size()
        # return torch.gather(word_rep[:, 1:, :], 1, orig_to_token_index.unsqueeze(-1).expand(batch_size, max_sent_len, rep_size))
