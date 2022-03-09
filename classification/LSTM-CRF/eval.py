from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import sys

states = ['abstract', 'strength', 'weakness', 'rating_summary', 
            'ac_disagreement', 'rebuttal_process', 'suggestion', 'decision', 'O']
correct_dict = {key: 0 for key in states}
wrong_dict = {key: 0 for key in states}

if len(sys.argv) == 2:
    pred_path = sys.argv[1]
else:
    print("[USAGE]: python eval_rouge.py <generated file>")

print("evaluating: ", pred_path)
# model="roberta-base-BiLSTM_encoder"
# output = open('results/'+model+'.results','r', encoding='utf-8').readlines()
output = open(pred_path,'r', encoding='utf-8').readlines()
correct = 0
wrong = 0
gold_list = []
pred_list = []
for line in output:
	if line.strip()!='':
		gold = line.strip().split('\t')[-2].split('-')[-1]
		pred = line.strip().split('\t')[-1].split('-')[-1]
		if gold==pred:
			correct += 1
			correct_dict[pred]+=1
		else:
			wrong += 1
			wrong_dict[pred]+=1

		gold_list.append(gold)
		pred_list.append(pred)


print(correct_dict)
print(wrong_dict)

print("Accuracy: ", 1.0*correct/(correct+wrong), '\tCorrect: ', correct, '\tWrong: ', wrong)

print('Macro F1: ', f1_score(gold_list, pred_list, average='macro'))

print('Micro F1: ', f1_score(gold_list, pred_list, average='micro'))

print(classification_report(gold_list, pred_list, digits=4))
# print('each class: ', f1_score(gold_list, pred_list, average=None))