from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


parser = ArgumentParser()
parser.add_argument("--pred_file", type=str, )
parser.add_argument("--ref_file", type=str, default='data/MedQA_seq2seq/test.ref')
args = parser.parse_args()

with open(args.pred_file, 'r') as f:
    pred_list = f.readlines()
pred_list = [pred.strip().split() for pred in pred_list]
pred_list = [list(filter(lambda p: p[0] != '<', pred)) for pred in pred_list]

with open(args.ref_file, 'r') as f:
    ref_list = f.readlines()
ref_list = [ref.strip().split() for ref in ref_list]
ref_list = [list(filter(lambda r: r[0] != '<', ref)) for ref in ref_list]

pair_hit = 0
uni_hit = 0
pair_cnt_pred, pair_cnt_ref = 0, 0
uni_cnt_pred, uni_cnt_ref = 0, 0

def match(pred, ref, pair_no):
    ref_index = [i for i, val in enumerate(ref) if val == 'Q%d' % pair_no] + [i for i, val in enumerate(ref) if val == 'A%d' % pair_no]
    q_ref_index = ref_index[0]
    pred_pair_no = int(pred[q_ref_index][1:]) if pred[q_ref_index].startswith('Q') else -1
    pred_index = [i for i, val in enumerate(pred) if val == 'Q%d' % pred_pair_no] + [i for i, val in enumerate(pred) if val == 'A%d' % pred_pair_no]
    return pred_index, ref_index

for pred, ref in zip(pred_list, ref_list):
    if len(pred) != len(ref):
        print(' '.join(pred))
        print(' '.join(ref))
        continue
    for pair_no in range(1, 111, 1):

        if 'Q%d' % pair_no not in ref:
            break
        pair_cnt_ref += 1
        pred_index, ref_index = match(pred, ref, pair_no)
        pair_hit += 1 if pred_index == ref_index else 0

        uni_hit += len(set(pred_index).intersection(ref_index))
        uni_cnt_pred += len(pred_index)
        uni_cnt_ref += len(ref_index)

    for pair_no in range(1, 111, 1):
        if 'Q%d' % pair_no in pred:
            pair_cnt_pred += 1    

precision = uni_hit / uni_cnt_pred
recall = uni_hit / uni_cnt_ref
f1 = 2 * precision * recall / (precision + recall)

adoption_rate = pair_hit / pair_cnt_pred
hit_rate = pair_hit / pair_cnt_ref
session_f1 = 2 * hit_rate * adoption_rate / (hit_rate + adoption_rate)

# ref_list = [r for ref in ref_list for r in ref]
# pred_list = [p for pred in pred_list for p in pred]
# precision, recall, f1, _ = precision_recall_fscore_support(ref_list, pred_list, average='micro', labels=list(set(ref_list) - set('O')))
print(args.pred_file, precision, recall, f1, adoption_rate, hit_rate, session_f1, sep='\t')
# print(classification_report(y_true=ref_list, y_pred=pred_list, labels=list(set(ref_list) - set('O'))))