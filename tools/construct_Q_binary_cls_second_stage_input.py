from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument("--pred_file", type=str, default='model/MedQA_seq2seq_Q_binary_cls/chinese-roberta-wwm-ext-large/predict_results.txt')
parser.add_argument("--test_file", type=str, default='data/MedQA_seq2seq/test.json')
parser.add_argument("--output_file", type=str, default='data/MedQA_seq2seq_A/test_MedQA_seq2seq_Q_binary_cls_chinese-roberta-wwm-ext-large.json')

args = parser.parse_args()

with open(args.test_file, 'r') as f:
    instance_list = f.readlines()
instance_list = [json.loads(instance) for instance in instance_list]

with open(args.pred_file, 'r') as f:
    pred_list = f.readlines()

fw = open(args.output_file, 'w', encoding='UTF-8')

for instance in instance_list:
    text_input, text_output = instance['text'], instance['summary']
    text_output += ' '
    current_infill_id = 0
    q_id = 1
    for i in range(len(text_output.split()) // 2):
        old_infill_token = '<extra_id_%d>' % i
        label = pred_list.pop(0).strip()
        if label == 'Q':
            label += str(q_id)
            q_id += 1
            text_input = text_input.replace(' ' + old_infill_token + ' ', label)
            text_output = text_output.replace('%s %s ' % (old_infill_token, label), '')
        else:
            new_infill_token = '<extra_id_%d>' % current_infill_id
            text_input = text_input.replace(old_infill_token, new_infill_token)
            text_output = text_output.replace(old_infill_token, new_infill_token)
            current_infill_id += 1

    # if '<extra_id_0>' in text_output:
    text_output = text_output.strip()
    if text_output == '':
        text_output = ' '
    print(
        json.dumps({'text': text_input, 'summary': text_output}, ensure_ascii=False),
        file=fw
    )