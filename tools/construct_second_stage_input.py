from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument("--pred_file", type=str, default='model/MedQA_seq2seq_Q/mt5-xl/generated_predictions.txt')
parser.add_argument("--test_file", type=str, default='data/MedQA_seq2seq/test.json')
parser.add_argument("--output_file", type=str, default='data/MedQA_seq2seq_A/test_MedQA_seq2seq_Q_mt5-xl.json')

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
    text_output += ' '
    current_infill_id = 0
    pred = pred[:len(text_output.split())]
    for i, label in enumerate(pred):
        old_infill_token = '<extra_id_%d>' % (i // 2)
        if label[0] in ['Q']:
            text_input = text_input.replace(' ' + pred[i-1] + ' ', label)
            text_output = text_output.replace('%s %s ' % (old_infill_token, label), '')
        elif label[0] in ['O']:
            new_infill_token = '<extra_id_%d>' % current_infill_id
            text_input = text_input.replace(old_infill_token, new_infill_token)
            text_output = text_output.replace(old_infill_token, new_infill_token)
            current_infill_id += 1

    # if '<extra_id_0>' in text_output:
    text_output = text_output.strip()
    if text_output == '':
        text_output = 'O'
    print(
        json.dumps({'text': text_input, 'summary': text_output}, ensure_ascii=False),
        file=fw
    )