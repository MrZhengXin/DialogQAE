import json
from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained('google/mt5-xl')

output_dir = '../MedQA_seq2seq/'

input_file_list = ['test-200.json', 'dev-100.json', 'train-700.json']

def can_cut(last_label_num, remain_label):
    for label_j in remain_label:
        label_j_num = int(label_j[1:]) if label_j != 'O' else 999
        if last_label_num >= label_j_num:
            return False
    return True

cut_len, cut_turn = 256, 20
max_length = 0
for input_file in input_file_list:
    with open(input_file, 'r') as f:
        instance_list = f.readline()
    instance_list = json.loads(instance_list)['data']
    fw = open(output_dir + input_file, 'w', encoding='UTF-8')
    for instance in instance_list:
        sent, role, label = instance['sent'], instance['role'], instance['label']
        sent = [s.replace(' ', '') for s in sent]
        len_dialog = len(sent)
        text_input, text_output = '', ''
        previous_infill_num = 0
        previous_label_num = 0
        max_label_num = 0
        label_num = 0
        for i in range(len_dialog):
            infill_token = '<extra_id_%d>' % (i - previous_infill_num)
            text_input += '%s: %s( %s ); ' % (role[i], sent[i], infill_token)
            label_num = int(label[i][1:]) if label[i] != 'O' else label_num
            max_label_num = max(max_label_num, label_num)
            label[i] = label[i][0] + str(label_num - previous_label_num) if label[i] != 'O' else 'O'
            text_output += infill_token + ' ' + label[i] + ' '
            
            if len(text_input) > cut_len or i - previous_infill_num > cut_turn:
                if not can_cut(max_label_num, label[i+1:]):
                    continue
                if len(t.encode(text_input)) < 512:
                    print(
                        json.dumps({'text': text_input.strip(), 'summary': text_output.strip()}, ensure_ascii=False),
                        file=fw
                    )
                    max_length = max(max_length, len(t.encode(text_input)))
                else:
                    print('1')
                text_input, text_output = '', ''
                previous_label_num = max_label_num
                previous_infill_num = i+1
        if text_input != '' :
            if len(t.encode(text_input)) < 512:
                max_length = max(max_length, len(t.encode(text_input)))
                print(
                    json.dumps({'text': text_input.strip(), 'summary': text_output.strip()}, ensure_ascii=False),
                    file=fw
                )
            else:
                print('1')
print(max_length)

