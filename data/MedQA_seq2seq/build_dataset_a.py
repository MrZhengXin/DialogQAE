import json
import re

output_dir = '../MedQA_seq2seq_A/'
input_file_list = ['test.json', 'dev.json', 'train.json']

for input_file in input_file_list:
    with open(input_file, 'r') as f:
        instance_list = f.readlines()
    fw = open(output_dir + input_file, 'w', encoding='UTF-8')

    instance_list = [json.loads(instance) for instance in instance_list]
    for instance in instance_list:
        text_input, text_output = instance['text'], instance['summary']
        text_output += ' '
        label_list = text_output.split()
        current_infill_id = 0
        for i, label in enumerate(label_list):
            old_infill_token = '<extra_id_%d>' % (i // 2)
            if label[0] in ['Q']:
                text_input = text_input.replace(' ' + label_list[i-1] + ' ', label)
                text_output = text_output.replace('%s %s ' % (old_infill_token, label), '')
            elif label[0] in ['O', 'A']:
                new_infill_token = '<extra_id_%d>' % current_infill_id
                text_input = text_input.replace(old_infill_token, new_infill_token)
                text_output = text_output.replace(old_infill_token, new_infill_token)
                current_infill_id += 1

        # text_output = re.sub('Q[0-9]*', 'O', text_output)
        # if '<extra_id_0>' in text_output:
        print(
            json.dumps({'text': text_input, 'summary': text_output.strip()}, ensure_ascii=False),
            file=fw
        )


