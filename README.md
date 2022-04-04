
We provide the full MReD data and its related code.
For more details please refer to the paper: https://arxiv.org/abs/2110.07474

<!-- INFO -->
## 1. Data Information
The `full_data_info.txt` file contains detailed information for the train, validation and test data splits for both classification and summarization tasks.
Each line have 12 fields separated by tab refering to the following information for the same paper:
* **data split**: `val`, `test` or `train`
* **paper id**: `yyyy-id`, where `yyyy` is the year
* **year**: from 2018 to 2021, the year of the ICLR submission 
* **openreview id**: `Paperxxx`, the paper id given by the open review website, can be found in the website's official blind review section, 2nd row in the format of `ICLRyyyy Conference Paperxxx AnonReviewerx`
* **reviewers' rating score**: `Rx: x, Rx: x, ..., Rx: x.` which gives a summary of the score given by each reviewer according to the anonymous reviewer number on openreview website. Note that sometimes the reviewer number is not continuous on openreview (eg., there is only R2, R3, R4)
* **average rating score**: the average score from all reviewers
* **paper decision**: `Reject` or `Accept (Oral)`, `Accept (Poster)`, `Accept (Spotlight)`, `Accept (Workshop)`
* **title**: the title of the paper/submisison
* **link**: the link of the paper/submission on openreview
* **word count**: number of words in meta-review
* **sentence-level category sequence**: `[label1, label2, ... ]`, per-sentence label sequence for the meta-review where `label1` represents the category label for 1st sentence, `label2` for the 2nd sentence and so on
* **segment-level category sequence**: `[label1, label2, ... ]`, label sequence for the meta-review on segment level where `label1` represents the category label for 1st segment (the sentences of the same label), `label2` for the 2nd segment and so on


<!-- CLASSIFICATION -->
## 2. Classification
### a. Data
In the folder `classification/data/`, we provide the human-annotated category label for every sentence in all meta-reviews. Each line contains three fields, namely **paper id**, **sentence**, and **label**, separated by tab. Consecutive meta-reviews are separated by an additional new line. You may search the **paper id** in `full_data_info.txt` for more information on a particular paper/submission.

### b. Code
The code for training the classifier is adapted from: https://github.com/allanj/pytorch_neural_crf.
Using Roberta-base, our trained model has a classification accuracy of 85.83%.
We include the our modified code in folder `classification/LSTM-CRF`.
In `classification/LSTM-CRF/data/mred`, we have already adapted our meta-review classification data to the required IOBES tagging format.
To train your own tagger, please follow the original repo for environment setup, then run
`python trainer.py --model_folder <Location to save traind model> --device <CUDA number> --embedder_type roberta-base`.
All hyperparameters are already set in `trainer.py` as default.
The trained model can be located in `classification/LSTM-CRF/model_files/`.
Prediction results are save to `classification/LSTM-CRF/results/`. To obtain the micro and macro F1 scores, run `python eval.py <result_file>`.


<!-- SUMMARIZATION -->
## 3. Summarization
### Extractive summarization
#### a. Data
We provide the full data of each paper reviews and metareviews in the source folder `summarization/extractive/data/source` and target folder `summarization/extractive/data/target` respectively, where each line represents the review or the meta-review passage in correspondence. The details of each paper can be found in `full_data_info.txt` following the file order. 

The target folder contains the following:
* full target files `{train/val/test}.target`, where each line is a meta-review
* the filtered target files `filtered_{train/val/test}.target` for those with meta-review word count between 20-400, where each line is a meta-review
* the filtered test target low score (<=3) and high score (>=7) files `filtered_test_{low/high}_score.target`, where each line is a meta-review, the meta-review word count is also between 20-400
* the filtered sentence-level labels files `filtered_test_sentence_labels.txt`, `filtered_test_sentence_labels_low_score.txt`, and `filtered_test_sentence_labels_high_score.txt`, where each line is the gold label sequence for the test data split
* the filered per sentence labeled files `filtered_train_labeled_sentences.txt`, `filtered_train_labeled_sentences_low_score.txt`, `filtered_train_labeled_sentences_high_score.txt`, where each line is a labeled sentence and each meta-review is separated by a newline.

The source folder contains the inputs obtained from different linearization methods in folders 
* `concat/`
* `rate-concat/`
* `merge/`
* `rate-merge/`
 Please refer to our paper for details of linearization methods. Inside each source and target folder, we provide a full data version and a filtered data version, where the filtered papers contains meta-review lengths no less than 20 words and no more than 400 words (which can be easily obtained by filtering according to the `full_data_info.txt`'s **word count** field). We used the filtered data in our paper's experiments. 

Moreover, we provide the classifier labeled (85.83% accuracy) per-sentence category label for filtered source inputs in `summarization/extractive/data/labeled_source/`, which include:
* the full labeled source files `{train,val,test}_labeled_sentences.txt`
* the filtered paper for meta-review length between 20-400 `filtered_{train,val,test}_labeled_sentences.txt`
* the filtered low score (<=3) and high score (>=7) paper labeled source inputs: `filtered_{train,val,test}_labeled_sentences_low_score.txt` and `filtered_{train,val,test}_labeled_sentences_high_score.txt`. 

#### b. Code
We provide code for the three exractive models and the generic baselines in our paper under the directory  `summarization/extractive/code/`.
* **Lexrank**: in folder `lexrank_model/`, based on the package: https://pypi.org/project/lexrank/. We extract the top n most relevant items from src, where n is decided according to the actual number of sentences in gold. The vanilla and labeled-controlled extractive summarization scripts are `lexrank_base.py` and `lexrank_control.py`. Need to run `pip install lexrank`.
* **Textrank**: in folder `textrank_model/`,based on the gensim.summarization.summarizer https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html. We use textrank to extract the top n most relevant items from src, where n is decided according to the actual number of sentences in gold. The vanilla and labeled-controlled extractive summarization scripts are `textrank_base.py` and `textrank_control.py`. Need to run `pip install gensim==3.8.3`.
* **MMR**: in folder `mmr_model/`,based on https://github.com/vishnu45/NLP-Extractive-NEWS-summarization-using-MMR/blob/master/mmr_summarizer.py. The vanilla and labeled-controlled extractive summarization scripts are `mmr_base.py` and `mmr_control.py`. Need to run `pip install gensim==3.8.3`.
* **Generic sentence baselines**: In folder `generic_baseline/`, the script `generic.py` uses textrank to first split all sentences in the input file (src/tgt, high/low papers...) into same label groups, arrange sentences in each group in order and then select according to meta-review label sequence. 
Usage: `python generic.py <input file> <label file> <name>` where name can be `src_generic`, `src_low_score`, `src_high_score`, `tgt_generic`, `tgt_low_score`, `tgt_high_score`.
Examples:
For `Source Low Score`: `python generic.py ../../data/labeled_source/filtered_train_labeled_sentences_low_score.txt ../../data/target/filtered_test_sentence_labels_low_score.txt src_low_score`
For `Target Low Score`: `python generic.py ../../data/target/filtered_train_labeled_sentences_low_score.txt ../../data/target/filtered_test_sentence_labels_low_score.txt tgt_low_score`

For the `Source Generic` baseline (see our paper for details), the input file `../../data/labeled_source/filtered_train_labeled_sentences.txt` may take exceedingly long time to be processed, and a faster method is to process it in chunks, `python generic.py ../../data/labeled_source/filtered_train_labeled_sentences.txt ../../data/target/filtered_test_sentence_labels.txt <chunk id> src_generic` by using chunk id from 0 to 8, then running `python comb_pt.py ../../data/target/filtered_test_sentence_labels.txt 9 src_generic` to combine the processed chunks to produce the output file.

The rouge score evaluation script is included as `eval_rouge.py`. Use `python eval_rouge.py <generated_file> <gold_file>` to run.

The script for linearization to obtain the `merge/` data is also included as `merge.py`.

### Abstractive summarization
#### a. Data
For vanilla generations, we include data in `summarization/absractive/filtered_uncontrolled_data/` with names according to different source linearizaton methods:
* concat: `{train, val, test}_concat.csv`
* rate-concat: `{train, val, test}_rate_concat.csv`
* merge: `{train, val, test}_merge.csv`
* rate-merge: `{train, val, test}_rate_concat.csv`
* longest review: `{train, val, test}_longest.csv`

For label controlled generation, we include data in `summarization/absractive/filtered_controlled_data/` according to different source linearization method similar to vanilla generation, and according to control methods:
* sentence control: file ending with `_sent-ctrl.csv`
* segment control: file ending with `_seg-ctrl.csv`


#### b. Code
We use the example code in Transformers https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py. 
Transformers version: "4.6.0.dev0"
please install datasets with version 1.6.2
Simply download the code from the Transformers and follow the installation instructions. When running the training script, you need to:
* specify the data file path by `--train_file <train_file_path> --validation_file <val_file_path> --test_file <test_file_path>`
* specify the maximum and minimun target generation length (if using filtered files) by `--max_target_length 400 --gen_target_min 20`
* specify the seed using `--seed <number>`
* specify source truncation using `--max_source_length <number>`. Note default transformers only supports up to 1024 tokens.

example usage `CUDA_VISIBLE_DEVICES=0 python run_summarization.py --model_name_or_path facebook/bart-large-cnn --do_train --do_eval --do_predict --train_file filtered_controlled_data/train_rate_concat_sent-ctrl.csv --validation_file filtered_controlled_data/val_rate_concat_sent-ctrl.csv --test_file filtered_controlled_data/test_rate_concat_sent-ctrl.csv --output_dir results/rate_concat_1024_sent-ctrl --overwrite_output_dir --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --predict_with_generate --seed 0 --save_total_limit 1 --max_source_length 1024 --max_target_length 400 --gen_target_min 20`

We have modified the model file to extend the input truncation for BART to more than 1024 tokens (default). You may use our modified file in `summarization/absractive/modeling_utils.py` to replace the file https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py if you wish to read more than 1024 tokens in the source and specify the truncation length by `--max_source_length <length>` when running the training script. The specific section we modified are commented with `# MReD` in `modeling_utils.py`.

## Cite Us
```
@article{shen2021mred,
  title={MReD: A Meta-Review Dataset for Controllable Text Generation},
  author={Shen, Chenhui and Cheng, Liying and Zhou, Ran and Bing, Lidong and You, Yang and Si, Luo},
  journal={arXiv preprint arXiv:2110.07474},
  year={2021}
}
```
