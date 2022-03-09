import argparse
import random
import numpy as np
from config import Reader, Config, ContextEmb, lr_decay, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances
import time
from model import NNCRF, TransformersCRF
import torch
from typing import List
from common import Instance
from termcolor import colored
import os
from config.utils import load_elmo_vec
from config.transformers_util import tokenize_instance, get_huggingface_optimizer_and_scheduler
from config import context_models, get_metric
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
import torch.nn as nn


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2','cuda:3'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=False,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="mred")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="adamw", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=50, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=20, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--static_context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="static contextual word embedding, our old ways to incorporate ELMo and BERT.")

    parser.add_argument('--embedder_type', type=str, default="roberta-base",
                        choices=["normal"] + list(context_models.keys()),
                        help="normal means word embedding + char, otherwise you can use 'bert-base-cased' and so on")
    parser.add_argument('--parallel_embedder', type=int, default=0,
                        choices=[0, 1],
                        help="use parallel training for those (BERT) models in the transformers. Parallel on GPUs")


    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]):
    ### Data Processing Info
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config, train_insts)
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)


    if config.embedder_type == "normal":
        model = NNCRF(config)
        optimizer = get_optimizer(config, model)
        scheduler = None
    else:
        print(
            colored(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}", 'red'))
        print(colored(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.", 'red'))
        print(colored(f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications.", 'red'))
        model = TransformersCRF(config)
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=len(batched_data) * epoch,
                                                                       weight_decay=0.0,
                                                                       eps = 1e-8,
                                                                       warmup_step=0)
        print(colored(f"[Optimizer Info] Modify the optimizer info as you need.", 'red'))
        print(optimizer)

    model.to(config.device)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    # if os.path.exists("model_files/" + model_folder):
    #     raise FileExistsError(
    #         f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
    #         f"to avoid override.")
    model_path = f"model_files/{model_folder}/lstm_crf.m"
    config_path = f"model_files/{model_folder}/config.conf"
    res_path = f"{res_folder}/{model_folder}.results"
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    os.makedirs(f"model_files/{model_folder}", exist_ok= True) ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in tqdm(np.random.permutation(len(batched_data)), desc="--training batch", total=len(batched_data)):
            model.train()
            loss = model(**batched_data[index])
            epoch_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            if scheduler is not None:
                scheduler.step()
        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_path)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_path, test_insts)
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

    print("Archiving the best Model...")
    try:
        with tarfile.open(f"model_files/{model_folder}/{model_folder}.tar.gz", "w:gz") as tar:
            tar.add(f"model_files/{model_folder}", arcname=os.path.basename(model_folder))
        print("Finished archiving the models")
    except:
        print("tar file not generated due to some exception")

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_path, test_insts)


def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance], print_each_type_metric: bool = False):
    ## evaluation
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    batch_id = 0
    batch_size = config.batch_size
    with torch.no_grad():
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = model.decode(**batch)
            batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch["labels"].to(config.device), batch["num_sents"].to(config.device), config.idx2labels)
            p_dict += batch_p
            total_predict_dict += batch_predict
            total_entity_dict += batch_total
            batch_id += 1
    if print_each_type_metric:
        for key in total_entity_dict:
            precision_key, recall_key, fscore_key = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])
            print(f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}")

    total_p = sum(list(p_dict.values()))
    total_predict = sum(list(total_predict_dict.values()))
    total_entity = sum(list(total_entity_dict.values()))
    precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
    print(colored(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, F1: {fscore:.2f}", 'blue'), flush=True)


    return [precision, recall, fscore]


def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader()
    set_seed(opt, conf.seed)

    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.static_context_emb != ContextEmb.none:
        print('Loading the static ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.static_context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.static_context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.static_context_emb.name + ".vec", tests)

    # conf.use_iobes(trains + devs + tests)
    conf.build_label_idx(trains + devs + tests)
    '''
    #labels: 20
    label 2idx: {'<PAD>': 0, 'B-abstract': 1, 'I-abstract': 2, 'B-strength': 3, 'I-strength': 4, 'B-decision': 5, 'B-weakness': 6, 'I-weakness': 7, 'B-rating_summary': 8, 'B-rebuttal_process': 9, 'B-suggestion': 10, 'I-rebuttal_process': 11, 'O': 12, 'B-ac_disagreement': 13, 'I-ac_disagreement': 14, 'I-suggestion': 15, 'I-rating_summary': 16, 'I-decision': 17, '<START>': 18, '<STOP>': 19}
    idx2label: ['<PAD>', 'B-abstract', 'I-abstract', 'B-strength', 'I-strength', 'B-decision', 'B-weakness', 'I-weakness', 'B-rating_summary', 'B-rebuttal_process', 'B-suggestion', 'I-rebuttal_process', 'O', 'B-ac_disagreement', 'I-ac_disagreement', 'I-suggestion', 'I-rating_summary', 'I-decision', '<START>', '<STOP>']
    '''

    if conf.embedder_type == "normal":
        conf.build_word_idx(trains, devs, tests)
        conf.build_emb_table()

        conf.map_insts_ids(trains)
        conf.map_insts_ids(devs)
        conf.map_insts_ids(tests)
        print("[Data Info] num chars: " + str(conf.num_char))
        # print(str(conf.char2idx))
        print("[Data Info] num words: " + str(len(conf.word2idx)))
        # print(config.word2idx)
    else:
        """
        If we use the pretrained model from transformers
        we need to use the pretrained tokenizer
        """
        print(colored(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer", "red"))
        tokenize_instance(context_models[conf.embedder_type]["tokenizer"].from_pretrained(conf.embedder_type), trains + devs + tests, conf.label2idx)

    train_model(conf, conf.num_epochs, trains, devs, tests)


if __name__ == "__main__":
    main()
