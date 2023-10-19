from argparse import ArgumentParser
import json
import re

parser = ArgumentParser()
parser.add_argument("--pred_file", type=str, default='model/ketang_Q/mt5-xl/generated_predictions.txt')
parser.add_argument("--test_file", type=str, default='data/ketang_Q/test.json')
parser.add_argument("--output_file", type=str, default='data/ketang_A/test_ketang_Q_mt5-xl.json')

args = parser.parse_args()

with open(args.test_file, 'r') as f:
    instance_list = f.readlines()
instance_list = [json.loads(instance) for instance in instance_list]

with open(args.pred_file, 'r') as f:
    pred_list = f.readlines()

fw = open(args.output_file, 'w', encoding='UTF-8')

for instance, pred in zip(instance_list, pred_list):
    pred = pred.strip().split()
    text_input, text_output = instance['text'], instance['summary']
    current_infill_id = 0
    pred = pred[:len(text_output.split())]
    for i, label in enumerate(pred):
        old_infill_token = '<extra_id_%d>' % (i // 2)
        q_prompt = ' ' + pred[i-1] + ' Q?'
        text_input = text_input.replace(q_prompt, ' ' + label + ' ')
    utterance_list = re.split(r'\); ', text_input)
    cnt = 0
    for i in range(len(utterance_list)):
        if utterance_list[i].startswith('D:'):
            utterance_list[i] = utterance_list[i].replace('( O ', '( <extra_id_%d> A?' % cnt)
            cnt += 1
    text_input = '); '.join(utterance_list)
    print(
        json.dumps({'text': text_input, 'summary': 'O'}, ensure_ascii=False),
        file=fw
    )