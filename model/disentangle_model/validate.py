import math
import random
import json
import torch
import torch.nn as nn
from .model import Autoencoder

def pad(list, padding=0, min_len=None):
    padded = []
    max_len = max([len(l) for l in list])
    if min_len:
        max_len = max(min_len, max_len)
    for l in list:
        padded.append(l + [padding] * (max_len - len(l)))

    return torch.tensor(padded, dtype=torch.long)

def EntropyLoss(tensor):
    '''

    :param tensor: (batch_size x m)
    :return:
    '''

    p = nn.functional.softmax(tensor, dim=1)
    log_p = torch.log(p)

    return - torch.mean(torch.sum(p * log_p, dim=1))

def accuracy(predictions, truths):
    pads = torch.Tensor.double(truths != 0)
    corrects = torch.Tensor.double(predictions == truths)
    valid_corrects = corrects * pads

    return valid_corrects.sum() / pads.sum()

def eval_loss(dists, text, pad_token):
    loss = 0
    num_tokens = 0

    for dist, y in zip(dists, text):
        y_len = sum([1 if y_i != pad_token else 0 for y_i in y])
        for i in range(y_len):
            loss -= torch.log(dist[i][y[i]])
            num_tokens += 1

    return loss, num_tokens

device = torch.device("cuda")

opt = {'emb_size': 300, 'hidden_size': 1000, 'unbias_size': 200, 'content_size': 800, 'rnn_class': nn.GRU, 'dropout': 0.0}

print(opt)

model = Autoencoder(emb_size=opt['emb_size'], hidden_size=opt['hidden_size'], unbias_size=opt['unbias_size'], content_size=opt['content_size'],
                    dict_file='/mnt/home/liuhaoc1/original_ParlAI_twitter_seq2seq_model/twitter_seq2seq_model.dict',
                    dropout=opt['dropout'], rnn_class=opt['rnn_class'], device=device).to(device)

model_file = 'save_model/model_with_bow.pt'
model.load_state_dict(torch.load(model_file))
print("load model from {}".format(model_file))

n_epoch = 20
batch_size = 30
lr = 0.001
clip = 0.25
k0 = 1
k1 = 1
k2 = 5

null_id = model.dict.tok2ind[model.dict.null_token]
eos_id = model.dict.tok2ind[model.dict.end_token]
unk_id = model.dict.tok2ind[model.dict.unk_token]
vocab_size = len(model.dict)


with open('../unbias_corpus.json', 'r') as f:
    data_list = json.load(f)
    valid_data_list = data_list[-5000:]

n = 10

valid_data = []
for i in range(len(valid_data_list) // (n * batch_size) + 1):
    nbatch = valid_data_list[i * (n * batch_size) : (i+1) * (n * batch_size)]
    nbatch_list = [([model.dict.tok2ind.get(word, unk_id) for word in ins[0].split()],
                    ins[1]) for ins in nbatch]
    descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)

    j = 0
    while len(descend_nbatch_list[j * batch_size : (j+1) * batch_size]) > 0:
        batch_list = descend_nbatch_list[j * batch_size : (j+1) * batch_size]

        # text: (batch_size x seq_len)
        text = pad([x[0] for x in batch_list], padding=null_id)
        labels = torch.tensor([x[1] for x in batch_list], dtype=torch.long)
        valid_data.append((text, labels))
        j += 1


model.eval()
total_loss = 0
total_num_tokens = 0
u_correct_num = 0.0
c_correct_num = 0.0
u_num = 0.0
c_num = 0.0
male_num = 0

r = random.randint(0, len(valid_data) - 1)

for i_batch, batch in enumerate(valid_data):
    # text: (batch_size x seq_len)
    # labels: (batch_size)
    text, labels = batch
    text = text.to(device)
    labels = labels.to(device)

    # preds: (batch_size x seq_len)
    # scores: (batch_size x seq_len x vocab_size)
    # preds, scores, y_u, y_c = model(text, train=False)
    # preds, scores, y_u, y_c = model(text, train=True)
    preds, scores, y_u_gender, y_c_gender, _, _ = model(text, train=True)
    y_u = y_u_gender
    y_c = y_c_gender

    dists = nn.functional.softmax(scores, dim=2)

    loss, num_tokens = eval_loss(dists, text, pad_token=null_id)
    total_loss += loss.item()
    total_num_tokens += num_tokens

    # re_acc = accuracy(preds, text)
    u_preds = y_u.argmax(dim=1)
    u_correct_num += torch.sum(u_preds == labels, dtype=torch.float32)
    u_num += len(u_preds)

    c_preds = y_c.argmax(dim=1)
    c_correct_num += torch.sum(c_preds == labels, dtype=torch.float32)
    c_num += len(c_preds)

    for truth, pred, u, c, label, c_score in zip(text, preds, u_preds, c_preds, labels, y_c):
        truth_null_id_list = [i for i in range(len(truth)) if truth[i] == null_id]
        pred_eos_id_list = [i for i in range(len(pred)) if pred[i] == eos_id]

        truth_null_id = len(truth) if len(truth_null_id_list) == 0 else truth_null_id_list[0]
        pred_eos_id = len(pred) if len(pred_eos_id_list) == 0 else pred_eos_id_list[0]

        print('truth: {}'.format(
            ' '.join([model.dict.ind2tok.get(idx, '__unk__') for idx in truth[:truth_null_id].tolist()])))
        print('pred: {}'.format(
            ' '.join([model.dict.ind2tok.get(idx, '__unk__') for idx in pred[:pred_eos_id].tolist()])))
        print("u: ", u.item())
        print("c: ", c.item())
        print("c_score: ", c_score.tolist())
        print("label: ", label.item())
        if label.item() == 0:
            male_num += 1
        print('------------------------------------------------------------')



ave_loss = total_loss / total_num_tokens
ppl = math.exp(ave_loss)

print("Validation ppl: {}  u_acc: {}  c_acc: {}".format(ppl, u_correct_num / u_num, c_correct_num / c_num))
print("male ratio: ", male_num / u_num)

print("Model performance: ppl_{:.4f} u_acc_{:.4f} c_acc_{:.4f}".format(ppl, u_correct_num / u_num,
                                                                       c_correct_num / c_num))