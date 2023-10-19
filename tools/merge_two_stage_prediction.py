from argparse import ArgumentParser
import json
import re

parser = ArgumentParser()
parser.add_argument("--pred_file", type=str, default='model/MedQA_seq2seq_A/mt5-xl/generated_predictions.txt')
parser.add_argument("--test_file", type=str, default='data/MedQA_seq2seq_A/test.json')
parser.add_argument("--output_file", type=str, default='model/MedQA_seq2seq_A/mt5-xl/generated_predictions_merged.txt')

args = parser.parse_args()

with open(args.test_file, 'r') as f:
    instance_list = f.readlines()
instance_list = [json.loads(instance) for instance in instance_list]

with open(args.pred_file, 'r') as f:
    pred_list = f.readlines()

fw = open(args.output_file, 'w', encoding='UTF-8')

for instance, pred in zip(instance_list, pred_list):
    pred = pred.strip().split()
    text_input = instance['text']
    for i, label in enumerate(pred):
        if label[0] != '<':
            text_input = text_input.replace(' ' + pred[i-1] + ' ', label).replace('A?);', ');')
    all_label_list = re.findall(r'\( ?[AOQ][0-9]* ?\);', text_input)
    all_label_list = [label.replace('(', '').replace(');', '').strip() for label in all_label_list]
    all_label = ' '.join(all_label_list) #.replace('(', '').replace(')', '').replace(';', '')
    print(all_label, file=fw)