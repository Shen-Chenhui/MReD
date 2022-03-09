from datasets import load_metric
import nltk
import sys

def get_file_content(file_path):
    content_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            processed_line = line.strip()
            content_list.append(processed_line)
    return content_list

metric = load_metric("./rouge")

if len(sys.argv) == 3:
    generated_file = sys.argv[1]
    gold_file = sys.argv[2]
    generated_list = get_file_content(generated_file)
    gold_list = get_file_content(gold_file)
    assert len(generated_list) == len(gold_list)
    result = metric.compute(predictions=generated_list, references=gold_list, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    print(result)
else:
    print("[USAGE]: python eval_rouge.py <generated_file> <gold_file>")