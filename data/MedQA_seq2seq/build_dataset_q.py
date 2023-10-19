import json
import re

output_dir = '../MedQA_seq2seq_Q/'
input_file_list = ['test.json', 'dev.json', 'train.json']

for input_file in input_file_list:
    with open(input_file, 'r') as f:
        instance_list = f.readlines()
    fw = open(output_dir + input_file, 'w', encoding='UTF-8')

    instance_list = [json.loads(instance) for instance in instance_list]
    for instance in instance_list:
        text_input, text_output = instance['text'], instance['summary']
        text_output = re.sub('A[0-9]*', 'O', text_output)
        print(
            json.dumps({'text': text_input, 'summary': text_output}, ensure_ascii=False),
            file=fw
        )


